"""Fig. 2, retrieval: keep layers 0..cut and remove every later layer (mixer and MLP), for every cut.
Metrics: choice accuracy among the 8 values over chat8/chat16/list8/rev8/long512, WikiText-2 perplexity.

    python -m experiments.truncation --model 9B [--shard 1/2]; ... --merge
"""
import json, argparse, time, glob
import numpy as np
from linquery.config import out_path
from linquery.runner import Runner
from linquery.evals import truncate, DictSuite, WikiPPL

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="9B")
ap.add_argument("--shard", default="1/1")
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
OUT = out_path("truncation", f"{a.model}.json")
SHARD = lambda k, n: OUT.with_name(f"{a.model}_shard{k}of{n}.json")

if a.merge:
    fs = sorted(glob.glob(str(OUT.with_name(f"{a.model}_shard*.json"))))
    rows = sorted((r for f in fs for r in json.load(open(f))["rows"]), key=lambda r: r["kept"])
    json.dump(dict(meta=json.load(open(fs[0]))["meta"], rows=rows), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(rows)} configurations")
    raise SystemExit

t0 = time.time()
R = Runner(a.model)
TEST, PPL = DictSuite(R), WikiPPL(R)
k, n = map(int, a.shard.split("/"))
res = dict(meta=dict(model=a.model, n_layers=R.n_layers, attn=R.attn_layers, linear=R.gdn_layers), rows=[])
for c in (list(range(R.n_layers - 1)) + [None])[k - 1::n]:
    h = truncate(R, c)
    d = TEST.run(h)
    r = dict(cut=c, kept=R.n_layers if c is None else c + 1, dict=d,
             dict_cmean=float(np.mean([v["cacc"] for v in d.values()])), ppl=PPL.run(h))
    res["rows"].append(r)
    print(f"[{(time.time() - t0) / 60:5.1f}m] kept {r['kept']:2d}  cacc {r['dict_cmean']:.3f}  ppl {r['ppl']:.1f}", flush=True)
    json.dump(res, open(SHARD(k, n), "w"), indent=1)
