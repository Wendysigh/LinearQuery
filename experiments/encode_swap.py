"""Table 1, left: token-level swaps that test what the Encode block contributes.

  value swap  host: entry t's value swapped with entry d's; donor: unswapped. Patch at t's value
              token. Success: the host answers v_t to the question about key t.
  key swap    host: unswapped, asks key t; donor: the keys of entries t and d exchanged. Patch at the
              key tokens of t and d. Success: the host answers v_d.
Patched: the Encode block's output, the input embedding, or another block's output (control).

    python -m experiments.encode_swap --model 9B
"""
import json, argparse, time, random
import numpy as np
import torch
from linquery.config import group_size, out_path
from linquery.runner import Runner, hook_ctx
from linquery.xfam import VALUES, KEYS, render
from linquery.prompts import chat_wrap

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="9B")
ap.add_argument("--n_test", type=int, default=200)
a = ap.parse_args()
OUT = out_path("encode", f"swap_{a.model}.json")
t0 = time.time()
Y = Runner(a.model)
BLOCKS = [G for G in Y.groups(group_size(a.model)) if G]
ENC = BLOCKS[0]
HY = Y.cfg.hidden_size

rng = random.Random(0)
keys = [k for k in KEYS if k not in VALUES]; rng.shuffle(keys)
vals = list(VALUES); rng.shuffle(vals)
K_TEST = keys[len(keys) // 2:]
V_TEST = vals[len(vals) // 2:]


def spans(pairs, q):
    """ids, per-entry key-token position and value-token position."""
    s, es, _ = render(chat_wrap(Y.tok), pairs, q, "chat")
    e = Y.tok(s, return_offsets_mapping=True)
    tokpos = lambda a_, b_: [i for i, (x, y) in enumerate(e.offset_mapping) if x < b_ and y > a_]
    kp, vp = [], []
    for (e0, e1), (k, v) in zip(es, pairs):
        seg = s[e0:e1]
        k0 = e0 + seg.index(k); v0 = e0 + seg.rindex(v)
        kp.append(tokpos(k0, k0 + len(k))); vp.append(tokpos(v0, v0 + len(v)))
    if any(len(x) != 1 for x in kp + vp):
        return None
    return e.input_ids, [x[0] for x in kp], [x[0] for x in vp]


def make_test(n):
    out = []
    cfgs = [(2, 5), (5, 2), (1, 6), (6, 1)]
    while len(out) < n:
        t, d = cfgs[len(out) % 4]
        ks = rng.sample(K_TEST, 8); vs = rng.sample(V_TEST, 8)
        base = list(zip(ks, vs))
        vsw = list(base); vsw[t] = (ks[t], vs[d]); vsw[d] = (ks[d], vs[t])     # value-swapped
        ksw = list(base); ksw[t] = (ks[d], vs[t]); ksw[d] = (ks[t], vs[d])     # key-swapped
        A = spans(base, ks[t]); V = spans(vsw, ks[t]); K = spans(ksw, ks[t])
        if not (A and V and K) or not (len(A[0]) == len(V[0]) == len(K[0])):
            continue
        aid = [Y.tok(v).input_ids for v in (vs[t], vs[d])]
        if all(len(x) == 1 for x in aid):
            out.append(dict(t=t, d=d, y=dict(base=A, vswap=V, kswap=K), id_t=aid[0][0], id_d=aid[1][0]))
    return out


class Writes:
    """Capture each layer's mixer output of `block` (all positions)."""
    def __init__(self, block): self.block, self.v = block, {}
    def register(self):
        hs = []
        for L in self.block:
            def fn(m, a_, o, L=L):
                self.v[L] = (o[0] if isinstance(o, tuple) else o).detach().float()
            hs.append(Y.mixer(L).register_forward_hook(fn))
        return _Handles(hs)


class SetWrites:
    """Overwrite each layer's mixer output of `block` at `pos` with vec[:, pos, i*H:(i+1)*H]."""
    def __init__(self, block, pos, vec): self.block, self.pos, self.vec = block, pos, vec
    def register(self):
        hs = []
        for i, L in enumerate(self.block):
            def fn(m, a_, o, i=i):
                t = (o[0] if isinstance(o, tuple) else o).clone()
                for p in self.pos:
                    t[:, p] = self.vec[:, p, i * HY:(i + 1) * HY].to(t.dtype).to(t.device)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            hs.append(Y.mixer(L).register_forward_hook(fn))
        return _Handles(hs)


class _Handles:
    def __init__(self, hs): self.hs = hs
    def remove(self):
        for h in self.hs:
            h.remove()


class SetEmb:
    """Overwrite the input embeddings at `pos` with the donor prompt's embeddings."""
    def __init__(self, pos, emb): self.pos, self.emb = pos, emb
    def register(self):
        def fn(m, a_, o):
            t = o.clone()
            for p in self.pos:
                t[:, p] = self.emb[:, p].to(t.dtype)
            return t
        return Y.model.get_input_embeddings().register_forward_hook(fn)


@torch.no_grad()
def block_writes(ids, block):
    w = Writes(block)
    with hook_ctx([w]):
        Y.model(torch.tensor(ids, device=Y.device), use_cache=False)
    return torch.cat([w.v[L] for L in block], -1)          # [B, T, len(block)*H]


test = make_test(a.n_test)
res = dict(meta=dict(model=a.model, encode=ENC, n_test=len(test)), swaps={})
EMB = Y.model.get_input_embeddings()


@torch.no_grad()
def host_run(ids, hooks, items):
    with hook_ctx(hooks):
        lg = Y.model(torch.tensor(ids, device=Y.device), use_cache=False).logits[:, -1].float()
    am = lg.argmax(-1)
    t_ = torch.tensor([i["id_t"] for i in items], device=lg.device)
    d_ = torch.tensor([i["id_d"] for i in items], device=lg.device)
    return (am == t_).float().tolist(), (am == d_).float().tolist()


def batches(swap):
    bk = {}
    for it in test:
        h = it["y"]["vswap" if swap == "value" else "base"]
        bk.setdefault((len(h[0]), tuple(h[1]), tuple(h[2]), it["t"]), []).append(it)
    return [v[i:i + 25] for v in bk.values() for i in range(0, len(v), 25)]


for swap in ("value", "key"):
    host_key, don_key = ("vswap", "base") if swap == "value" else ("base", "kswap")
    Rs = {}
    for bt in batches(swap):
        t, d = bt[0]["t"], bt[0]["d"]
        hy = [it["y"][host_key][0] for it in bt]; dy = [it["y"][don_key][0] for it in bt]
        pos = [bt[0]["y"][host_key][2][t]] if swap == "value" else [bt[0]["y"][host_key][1][t], bt[0]["y"][host_key][1][d]]

        def add(name, hooks):
            at, ad = host_run(hy, hooks, bt)
            r = Rs.setdefault(name, dict(success=[], own=[]))
            r["success"] += at if swap == "value" else ad
            r["own"] += ad if swap == "value" else at
        add("no patch", [])
        add("encode", [SetWrites(ENC, pos, block_writes(dy, ENC))])
        with torch.no_grad():
            emb = EMB(torch.tensor(dy, device=Y.device))
        add("embedding", [SetEmb(pos, emb)])
        for bi, G in enumerate(BLOCKS[1:], 1):
            add(f"G{bi}", [SetWrites(G, pos, block_writes(dy, G))])
    res["swaps"][swap] = {k: {m: float(np.mean(v)) for m, v in r.items()} for k, r in Rs.items()}
    for k, v in res["swaps"][swap].items():
        print(f"{swap:5s} | {k:12s} success {v['success']:.2f}  own {v['own']:.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT} ({(time.time() - t0) / 60:.1f} min)", flush=True)
