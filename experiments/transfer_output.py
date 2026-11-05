"""Table 2, output transplant: a linear map fitted on paired runs takes the donor's Address output at the
final token to the receiver's; f(donor asks A) is written into the receiver's run that asks B.

  fit   keys from K_fit only; every dictionary is built from K_fit
  test  keys from K_test only (disjoint): the map never saw any key in the test dictionaries

Conditions on the test pairs (receiver asks B; donor and receiver share the dictionary):
  in-family      the receiver's own write from its run asking A (the same-model baseline)
  xfam           f(donor Address, donor asks A)
  xfam-self      f(donor Address, donor asks B) (map fidelity: should keep B)
  xfam-shuffled  f fitted on shuffled pairs, donor asks A (control)
  src=<block>    f fitted from another donor block, donor asks A
  mean           the receiver's write replaced by its fit-set mean (removes the query)

    python -m experiments.transfer_output --donor 0.8B --recv G1B --format chat8
    python -m experiments.transfer_output --donor 9B --recv G7B --recv_query 16-24,26-34 --name G2G3
    python -m experiments.transfer_output --donor G7B --recv 9B --combine_src 2+3 --name G2G3
"""
import json, argparse, time, random
import numpy as np
import torch
from linquery.config import group_size, reader, out_path
from linquery.runner import Runner, hook_ctx
from linquery.qkv import AttnWeights
from linquery.xfam import KEYS, VALUES, encode

ADDRESS = {"0.8B": [12, 13, 14], "9B": [16, 17, 18], "G1B": list(range(16, 25)), "G7B": list(range(16, 25))}

ap = argparse.ArgumentParser()
ap.add_argument("--donor", default="0.8B")
ap.add_argument("--recv", default="G1B")
ap.add_argument("--format", default="chat8")
ap.add_argument("--n_fit", type=int, default=6000)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--name", default="", help="suffix of the output file")
ap.add_argument("--recv_query", default="", help="the receiver's target layers, e.g. 16-24,26-34")
ap.add_argument("--combine_src", default="", help="extra donor sources concatenating several donor blocks, e.g. 2+3")
a = ap.parse_args()
OUT = out_path("transfer", f"output_{a.donor}_to_{a.recv}_{a.format}" + (f"_{a.name}" if a.name else "") + ".json")
STYLE = {"list8": "list", "rev8": "rev"}.get(a.format, "chat")
t0 = time.time()


def log(*s):
    print(f"[{(time.time() - t0) / 60:6.1f}m]", *s, flush=True)


D = Runner(a.donor)                  # mamba_ssm kernels need both models on the current device
Y = Runner(a.recv)
D_BLOCKS = D.groups(group_size(a.donor))
DQ, YQ = ADDRESS[a.donor], ADDRESS[a.recv]
if a.recv_query:
    YQ = [L for part in a.recv_query.split(",") for L in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)]
RL, RH = reader(a.recv, "chat8")
log(f"donor {a.donor} blocks {D_BLOCKS} query {DQ} | receiver {a.recv} query {YQ} reader L{RL}H{RH}")

rng = random.Random(0)
keys = [k for k in KEYS if k not in VALUES]
rng.shuffle(keys)
K_FIT, K_TEST = keys[:len(keys) // 2], keys[len(keys) // 2:]


def make(keyset, n, with_pair):
    """items: dictionary of 8 pairs; asks A (and B when with_pair)."""
    out = []
    cfgs = [(2, 5), (5, 2), (1, 6), (6, 1)]
    while len(out) < n:
        ia, ib = cfgs[len(out) % 4] if with_pair else (rng.randrange(8), None)
        ks = rng.sample(keyset, 8); vs = rng.sample(VALUES, 8)
        pairs = list(zip(ks, vs))
        it = dict(pairs=pairs, ia=ia, ib=ib)
        for tag, M in (("d", D), ("y", Y)):
            it[f"{tag}A"], it[f"{tag}ent"], _ = encode(M.tok, pairs, ks[ia], STYLE)
            if with_pair:
                it[f"{tag}B"], _, _ = encode(M.tok, pairs, ks[ib], STYLE)
                if len(it[f"{tag}A"]) != len(it[f"{tag}B"]):
                    break
        else:
            aid, bid = Y.tok(vs[ia]).input_ids, (Y.tok(vs[ib]).input_ids if with_pair else [0])
            if len(aid) == 1 and len(bid) == 1:
                it["aid"], it["bid"] = aid[0], bid[0]
                out.append(it)
    return out


def buckets(items, keys_):
    bk = {}
    for it in items:
        bk.setdefault(tuple(len(it[k]) for k in keys_) + (str(it["yent"]),), []).append(it)
    return [v[i:i + 32] for v in bk.values() for i in range(0, len(v), 32)]


class Cap:
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(m, a_, o):
            self.v = (o[0] if isinstance(o, tuple) else o)[:, -1].detach().float().clone()
        return self.M.mixer(self.L).register_forward_hook(fn)


class SetFinal:
    """Overwrite a mixer's output at the final position with `vec` [B, H]."""
    def __init__(self, M, L, vec): self.M, self.L, self.vec = M, L, vec
    def register(self):
        def fn(m, a_, o):
            t = (o[0] if isinstance(o, tuple) else o).clone()
            t[:, -1] = self.vec.to(t.dtype).to(t.device)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return self.M.mixer(self.L).register_forward_hook(fn)


@torch.no_grad()
def writes(M, ids_list, layers):
    ids = torch.tensor(ids_list, device=M.device)
    caps = [Cap(M, L) for L in layers]
    with hook_ctx(caps):
        M.model(ids, use_cache=False)
    return torch.cat([c.v for c in caps], -1)            # [B, len(layers)*H]


# ------------------------------------------------------------------ fit data
D_ALL = sorted({L for G in D_BLOCKS for L in G})
EMB = D.model.get_input_embeddings()
HD = D.cfg.hidden_size


def kid(M, it, which):
    return M.tok(" " + it["pairs"][it[which]][0]).input_ids[0]


@torch.no_grad()
def donor_feats(bt, key):
    """{source: [B, dim]}: every donor block's write at the final token, and the donor's static
    input embedding of the queried key (no context at all)."""
    w = writes(D, [it["d" + key] for it in bt], D_ALL).cpu().double()
    per = {L: w[:, i * HD:(i + 1) * HD] for i, L in enumerate(D_ALL)}
    f = {f"G{bi}": torch.cat([per[L] for L in G], -1) for bi, G in enumerate(D_BLOCKS) if G}
    for spec in filter(None, a.combine_src.split(",")):
        idx = [int(i) for i in spec.split("+")]
        f["+".join(f"G{i}" for i in idx)] = torch.cat([f[f"G{i}"] for i in idx], -1)
    ids = torch.tensor([kid(D, it, "ia" if key == "A" else "ib") for it in bt], device=D.device)
    f["emb"] = EMB(ids).double().cpu()
    return f


fit = make(K_FIT, a.n_fit, with_pair=False)
Xf, Yf = {}, []
for bt in buckets(fit, ["dA", "yA"]):
    for k, v in donor_feats(bt, "A").items():
        Xf.setdefault(k, []).append(v)
    Yf.append(writes(Y, [it["yA"] for it in bt], YQ).cpu())
Xf = {k: torch.cat(v) for k, v in Xf.items()}
Yf = torch.cat(Yf).double()
SOURCES = list(Xf)
q_name = next(f"G{bi}" for bi, G in enumerate(D_BLOCKS) if G == DQ)
log(f"fit set: {len(Yf)} prompts, K_fit {len(K_FIT)} keys; donor dim {HD}, target dim {Yf.shape[1]}; "
    f"sources {SOURCES} (query block {q_name})")


def ridge(X, T):
    """Centred ridge, lambda chosen on a held-out fifth of the fit set (at most 500 prompts)."""
    n_val = min(500, max(5, len(X) // 5))
    Xt, Tt, Xv, Tv = X[n_val:], T[n_val:], X[:n_val].cuda(), T[:n_val].cuda()
    mx, mt = Xt.mean(0).cuda(), Tt.mean(0).cuda()
    Xc, Tc = Xt.cuda() - mx, Tt.cuda() - mt
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    UtT = U.T @ Tc
    best = None
    for p_ in np.arange(-3, 6.5, 0.5):
        lam = 10 ** p_ * float(S[0] ** 2) * 1e-6
        W = Vh.T @ (UtT * (S / (S ** 2 + lam))[:, None])
        err = float((((Xv - mx) @ W + mt - Tv) ** 2).sum() / ((Tv - mt) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, W)
    W = best[1]
    return (lambda Xn: (Xn.double().cuda() - mx) @ W + mt), 1 - best[0]


# ------------------------------------------------------------------ test data
test = make(K_TEST, a.n_test, with_pair=True)
TB = buckets(test, ["dA", "yA", "yB"])
HY = Y.cfg.hidden_size
FA = [donor_feats(bt, "A") for bt in TB]
FB = [donor_feats(bt, "B") for bt in TB]
YA = [writes(Y, [it["yA"] for it in bt], YQ) for bt in TB]


@torch.no_grad()
def recv_run(bt, vec=None):
    ids = torch.tensor([it["yB"] for it in bt], device=Y.device)
    hooks = []
    if vec is not None:
        hooks = [SetFinal(Y, L, vec[:, i * HY:(i + 1) * HY]) for i, L in enumerate(YQ)]
    aw = AttnWeights(Y.mixer(RL))
    with hook_ctx(hooks + [aw]):
        lg = Y.model(ids, use_cache=False).logits[:, -1].float()
    at = aw.value[:, RH, -1].float()
    ent = bt[0]["yent"]
    ea = torch.stack([at[j, ent[it["ia"]]].sum() for j, it in enumerate(bt)])
    eb = torch.stack([at[j, ent[it["ib"]]].sum() for j, it in enumerate(bt)])
    aid = torch.tensor([it["aid"] for it in bt], device=lg.device)
    bid = torch.tensor([it["bid"] for it in bt], device=lg.device)
    am = lg.argmax(-1)
    return dict(ansA=(am == aid).float().tolist(), ansB=(am == bid).float().tolist(),
                dAB=(lg.gather(1, aid[:, None]) - lg.gather(1, bid[:, None]))[:, 0].tolist(),
                attA=ea.tolist(), attB=eb.tolist())


def evaluate(vec_fn):
    """vec_fn(batch index) -> receiver query-block write [B, dim] or None; mean metrics."""
    acc = {}
    for i, bt in enumerate(TB):
        for k, v in recv_run(bt, vec_fn(i)).items():
            acc.setdefault(k, []).extend(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def test_r2(f, src):
    P = torch.cat([f(FA[i][src]).cpu() for i in range(len(TB))])
    T = torch.cat([y.double().cpu() for y in YA])
    return float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())


res = dict(meta=dict(donor=a.donor, recv=a.recv, format=a.format, donor_query=DQ, recv_query=YQ,
                     reader=[RL, RH], donor_blocks=D_BLOCKS, query_source=q_name, n_fit=len(Yf),
                     n_test=sum(len(b) for b in TB), k_fit=K_FIT, k_test=K_TEST))
C = res["conditions"] = {}
ymean = Yf.mean(0)
C["host (no transplant)"] = evaluate(lambda i: None)
C["in-family"] = evaluate(lambda i: YA[i])
C["mean"] = evaluate(lambda i: ymean.expand(len(TB[i]), -1))
maps = {s: ridge(Xf[s], Yf) for s in SOURCES}
perm = torch.randperm(len(Yf), generator=torch.Generator().manual_seed(1))
fsh, _ = ridge(Xf[q_name], Yf[perm])
fq = maps[q_name][0]
C["xfam"] = evaluate(lambda i: fq(FA[i][q_name]))
C["xfam-self"] = evaluate(lambda i: fq(FB[i][q_name]))
C["xfam-shuffled"] = evaluate(lambda i: fsh(FA[i][q_name]))
for s in SOURCES:
    if s != q_name:
        f = maps[s][0]
        C[f"src={s}"] = evaluate(lambda i, f=f, s=s: f(FA[i][s]))
res["r2"] = {s: dict(val=maps[s][1], test=test_r2(maps[s][0], s)) for s in SOURCES}
for k, c in C.items():
    log(f"{k:22s} ansA {c['ansA']:.2f}  ansB {c['ansB']:.2f}  D(A-B) {c['dAB']:+6.2f}  "
        f"attA {c['attA']:.2f}  attB {c['attB']:.2f}")
log("R^2 (val / test on unseen keys): " + "  ".join(
    f"{s}:{v['val']:.2f}/{v['test']:.2f}" for s, v in res["r2"].items()))
json.dump(res, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
