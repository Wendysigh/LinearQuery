"""Transplant: is a block's output at the final position sufficient to determine what is retrieved?

Donor run A and host run B share one dictionary and differ only in the queried key. A block's mixer
outputs from run A replace those of run B at the chosen positions. Reported: how often B answers A's
value (ansA), and each tracked reader's attention on entries A and B.

    python -m experiments.transplant --model 9B --format chat8 --positions final,dict
    python -m experiments.transplant --model 27B --format chat8 --union "8;9;8,11,12;8-12" --name union
    python -m experiments.transplant --model 9B --format chat8 --layers 16,17,18 --name layers
"""
import json, argparse, time, torch
import numpy as np
from linquery.config import MODELS, reader, group_size, out_path
from linquery.runner import Runner, hook_ctx, Capture, Transplant
from linquery.qkv import AttnWeights
from linquery.prompts import pair_batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--format", default="chat8")
ap.add_argument("--union", default="", help="block sets transplanted jointly, separated by ';', e.g. '8,11,12;8-12'")
ap.add_argument("--layers", default="", help="treat each listed layer as its own block, e.g. 16,17,18")
ap.add_argument("--positions", default="final", help="final, question, dict, all")
ap.add_argument("--name", default="", help="suffix of the output file")
a = ap.parse_args()
OUT = out_path("transplant", f"{a.model}_{a.format}" + (f"_{a.name}" if a.name else "") + ".json")

t0 = time.time()
R = Runner(a.model)
RL, RH = reader(a.model, a.format)
WATCH = [tuple(map(int, w.split(","))) for w in MODELS[a.model].get("watch", [])] if a.format == "chat8" else []


def parse_set(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-")); out += list(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


BASE = R.groups(group_size(a.model))
if a.layers:
    GROUPS = [[L] for L in map(int, a.layers.split(","))]
    NAMES = [f"L{G[0]}" for G in GROUPS]
elif a.union:
    SETS = [parse_set(x) for x in a.union.split(";")]
    GROUPS = [sorted(L for g in gs for L in BASE[g]) for gs in SETS]
    NAMES = ["+".join(f"G{g}" for g in gs) for gs in SETS]
else:
    GROUPS, NAMES = BASE, [f"G{i}" for i in range(len(BASE))]
print(f"[{a.model} {a.format}] {dict(zip(NAMES, GROUPS))}  reader L{RL}H{RH}  watch {WATCH}", flush=True)

B = pair_batches(R.tok, a.format, R.device)
N = sum(len(b["aid"]) for b in B)
print(f"  {N} prompt pairs in {len(B)} batches, {B[0]['n']} tokens", flush=True)


def pos_set(b, which):
    n, qs = b["n"], b["qstart"]
    return {"final": [n - 1], "question": list(range(qs, n)), "dict": list(range(0, qs)), "all": list(range(n))}[which]


@torch.no_grad()
def run(ids, b, hooks=()):
    """Final-position logits and, per tracked head (the reader first), attention on each entry."""
    aws = {L: AttnWeights(R.mixer(L)) for L in {RL} | {L for L, _ in WATCH}}
    with hook_ctx(list(hooks) + list(aws.values())):
        lg = R.model(ids, use_cache=False).logits[:, -1].float()
    ent = []
    for L, H in [(RL, RH)] + WATCH:
        at = aws[L].value[:, H, b["n"] - 1].float()
        ent.append(torch.stack([at[:, e].sum(-1) for e in b["entry"]], 1))
    return lg, ent


def score(lg, ent, b):
    am = lg.argmax(-1)
    return dict(ansA=float((am == b["aid"]).float().mean()),
                ansB=float((am == b["bid"]).float().mean()),
                dAB=float((lg.gather(1, b["aid"][:, None]) - lg.gather(1, b["bid"][:, None])).mean()),
                attA=float(ent[0][:, b["ia"]].mean()), attB=float(ent[0][:, b["ib"]].mean()),
                **{f"L{L}H{H}_{x}": float(e[:, b[i]].mean())
                   for (L, H), e in zip(WATCH, ent[1:]) for x, i in (("A", "ia"), ("B", "ib"))})


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


res = dict(meta=dict(model=a.model, format=a.format, reader=[RL, RH], watch=WATCH, n=N, tokens=B[0]["n"]))
base_a, base_b = [], []
for b in B:
    la, ea = run(b["ids_a"], b); base_a.append(score(la, ea, b))
    lb, eb = run(b["ids_b"], b); base_b.append(score(lb, eb, b))
res["baseline"] = dict(run_A=agg(base_a), run_B=agg(base_b))
print(f"run A: ansA {res['baseline']['run_A']['ansA']:.2f}  attA {res['baseline']['run_A']['attA']:.2f}   "
      f"run B: ansB {res['baseline']['run_B']['ansB']:.2f}  attB {res['baseline']['run_B']['attB']:.2f}", flush=True)

res["transplant"] = []
for gi, G in enumerate(GROUPS):
    if not G:
        continue
    for which in a.positions.split(","):
        rows = []
        for b in B:
            caps = [Capture(R, L) for L in G]
            with hook_ctx(caps):
                R.model(b["ids_a"], use_cache=False)          # donor = run A
            p = pos_set(b, which)
            lg, ent = run(b["ids_b"], b, [Transplant(R, L, p, c.value) for L, c in zip(G, caps)])
            rows.append(score(lg, ent, b))
        r = agg(rows); r.update(group=gi, name=NAMES[gi], layers=G, positions=which)
        res["transplant"].append(r)
        print(f"{NAMES[gi]:>20s} {which:6s} ansA {r['ansA']:.2f}  ansB {r['ansB']:.2f}  attA {r['attA']:.2f}"
              + "".join(f"  L{L}H{H} {r[f'L{L}H{H}_A']:.2f}" for L, H in WATCH), flush=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
