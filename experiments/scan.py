"""Removal scan of one (model, format).
  baseline   accuracy and logit gap
  blocks     remove each linear block (zero its mixer outputs)
  layers     remove each single linear layer
  readers    every softmax head's attention from the final token to the queried entry; of the top 8,
             the head whose removal lowers the gap most is the reader
With --qk_groups, for the head given by --reader (default: configs/models.yaml):
  selectivity  log attention mass on the queried entry minus the mean over other entries, with the
               query and the keys taken from the intact run or a run with the block removed
  reader_split the reader's attention on the queried entry with the block removed

    python -m experiments.scan --model 9B --format chat8 --qk_groups 0,1,2,3,4,5,6,7
"""
import json, argparse, time, torch
import numpy as np
from linquery.config import reader, group_size, out_path
from linquery.runner import Runner, hook_ctx, Zero
from linquery.qkv import AttnWeights, ProjPatch, ProjCapture
from linquery.prompts import make_items, batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--format", default="chat8")
ap.add_argument("--max_bs", type=int, default=25)
ap.add_argument("--reader", default="", help="head for selectivity and reader_split, e.g. 19,11")
ap.add_argument("--qk_groups", default="", help="block indices, e.g. 3,4")
a = ap.parse_args()
OUT = out_path("scan", f"{a.model}_{a.format}.json")

t0 = time.time()
R = Runner(a.model)
tok = R.tok
HD = R.head_dim
NQ = R.cfg.num_attention_heads
QGATE = R.q_gated
GROUPS = R.groups(group_size(a.model))
print(f"[{a.model} {a.format}] blocks {GROUPS}", flush=True)

B = batches(tok, make_items(a.format, tok), R.device, a.max_bs)
N = sum(len(b.items) for b in B)
print(f"  {N} items in {len(B)} batches, ~{B[0].pos['n']} tokens", flush=True)


@torch.no_grad()
def run_batch(ids, hooks=()):
    with hook_ctx(hooks):
        return R.model(ids, use_cache=False).logits[:, -1]


@torch.no_grad()
def evaluate(hooks_fn=lambda b: []):
    """Accuracy over clean and corrupted prompts, and the logit gap of each clean/corrupted pair."""
    ok = tot = 0; gaps = []
    for b in B:
        ds = []
        for ids, want in ((b.clean_ids, b.aid), (b.corr_ids, b.cid)):
            lg = run_batch(ids, hooks_fn(b))
            ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
            ds.append(b.D(lg))
        gaps += (ds[0] - ds[1]).tolist()
    return dict(acc=ok / tot, gap=float(np.mean(gaps)), gap_sem=float(np.std(gaps) / len(gaps) ** .5))


res = dict(meta=dict(model=a.model, format=a.format, n=N, tokens=B[0].pos["n"], groups=GROUPS))
res["baseline"] = evaluate()
print(f"baseline acc {res['baseline']['acc']:.3f} gap {res['baseline']['gap']:+.2f}", flush=True)

res["groups"] = []
for gi, G in enumerate(GROUPS):
    r = evaluate(lambda b, G=G: [Zero(R, L) for L in G]); r.update(group=gi, layers=G)
    res["groups"].append(r)
print("blocks  " + "  ".join(f"G{r['group']}:{r['acc']:.2f}/{r['gap']:+.1f}" for r in res["groups"]), flush=True)

res["layers"] = []
for L in R.gdn_layers:
    r = evaluate(lambda b, L=L: [Zero(R, L)]); r.update(layer=L)
    res["layers"].append(r)
print("layers  " + "  ".join(f"L{r['layer']}:{r['acc']:.2f}" for r in res["layers"]), flush=True)


@torch.no_grad()
def attn_split(layer, hooks_fn=lambda b: []):
    """Per head of `layer`: attention mass from the final token on the queried entry / other
    entries / first token, averaged over items."""
    acc = []
    for b in B:
        aw = AttnWeights(R.mixer(layer))
        with hook_ctx([aw] + hooks_fn(b)):
            R.model(b.clean_ids, use_cache=False)
        at = aw.value[:, :, b.pos["final"]].float()                 # [B, H, T]
        ep = b.pos["entry_pos"]
        tgt = at[:, :, ep[b.ti]].sum(-1)
        oth = sum(at[:, :, e].sum(-1) for j, e in enumerate(ep) if j != b.ti)
        acc.append(torch.stack([tgt, oth, at[:, :, 0]], -1).cpu())
    return torch.cat(acc).mean(0)                                    # [H, 3]


heads = []
for L in R.attn_layers:
    s = attn_split(L)
    for h in range(NQ):
        heads.append(dict(layer=L, head=h, target=float(s[h, 0]), others=float(s[h, 1]), first=float(s[h, 2])))
heads.sort(key=lambda r: -r["target"])


class AttnHeadZero:
    """Zero one softmax head's output (its slice of the o_proj input)."""
    def __init__(self, L, h): self.L, self.h = L, h
    def register(self):
        def fn(mod, args, kwargs):
            t = (args[0] if args else kwargs["input"]).clone()
            t[..., self.h * HD:(self.h + 1) * HD] = 0
            return (t,) + tuple(args[1:]), kwargs
        return R.mixer(self.L).o_proj.register_forward_pre_hook(fn, with_kwargs=True)


for r in heads[:8]:
    e = evaluate(lambda b, r=r: [AttnHeadZero(r["layer"], r["head"])])
    r.update(zero_acc=e["acc"], zero_gap=e["gap"])
res["reader_heads"] = heads[:10]
best = sorted(heads[:8], key=lambda r: r["zero_gap"])[0]
res["meta"]["reader_found"] = [best["layer"], best["head"]]
print("readers " + "  ".join(f"L{r['layer']}H{r['head']}:{r['target']:.2f}(zero gap {r['zero_gap']:+.1f})"
                              for r in heads[:8]) + f"  -> L{best['layer']}H{best['head']}", flush=True)

if a.qk_groups:
    RL, RH = map(int, a.reader.split(",")) if a.reader else reader(a.model, a.format)
    ATTN = R.mixer(RL)
    QK = [(f"G{i}", GROUPS[int(i)]) for i in a.qk_groups.split(",")]

    @torch.no_grad()
    def cap_qk(ids, zl):
        cq, ck = ProjCapture(ATTN, "q"), ProjCapture(ATTN, "k")
        with hook_ctx([cq, ck] + [Zero(R, L) for L in zl]):
            R.model(ids, use_cache=False)
        return cq.value, ck.value

    @torch.no_grad()
    def selectivity(qsrc, ksrc, DON):
        sel, top = [], []
        for bi, b in enumerate(B):
            n = b.pos["n"]; aw = AttnWeights(ATTN)
            hooks = [aw, ProjPatch(ATTN, "q", [n - 1], DON[(bi, qsrc)][0], HD, gated=QGATE),
                     ProjPatch(ATTN, "k", list(range(n)), DON[(bi, ksrc)][1], HD, gated=False)]
            with hook_ctx(hooks):
                R.model(b.clean_ids, use_cache=False)
            at = aw.value[:, RH, b.pos["final"]].float()
            s = torch.stack([at[:, e].sum(-1).clamp_min(1e-30).log() for e in b.pos["entry_pos"]], 1)
            t = s[:, b.ti]; o = torch.cat([s[:, :b.ti], s[:, b.ti + 1:]], 1)
            sel += (t - o.mean(1)).tolist()
            top += (t > o.max(1).values).float().tolist()
        return dict(selectivity=float(np.mean(sel)), top1=float(np.mean(top)))

    res["selectivity"] = []
    for name, G in QK:
        DON = {}
        for bi, b in enumerate(B):
            DON[(bi, "intact")] = cap_qk(b.clean_ids, [])
            DON[(bi, name)] = cap_qk(b.clean_ids, G)
        for qs in ("intact", name):
            for ks in ("intact", name):
                r = selectivity(qs, ks, DON); r.update(group=name, layers=G, q=qs, k=ks)
                res["selectivity"].append(r)
        del DON
    res["reader_split"] = {}
    for name, G in [("intact", [])] + QK:
        s = attn_split(RL, lambda b, G=G: [Zero(R, L) for L in G])[RH]
        res["reader_split"][name] = dict(target=float(s[0]), others=float(s[1]), first=float(s[2]))
    res["meta"].update(reader=[RL, RH])
    print("split   " + "  ".join(f"{k}:{v['target']:.2f}" for k, v in res["reader_split"].items()), flush=True)

json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
