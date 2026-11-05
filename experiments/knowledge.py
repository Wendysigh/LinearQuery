"""Fig. 2, knowledge: six lm-evaluation-harness tasks (zero-shot, 500 items each) along the truncation curve.

    python -m experiments.knowledge --model 9B [--shard 1/4]; ... --merge
"""
import json, argparse, time, glob
from linquery.config import out_path
from linquery.runner import Runner, hook_ctx
from linquery.evals import truncate

TASKS = ["arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa", "hellaswag"]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="9B")
ap.add_argument("--shard", default="1/1")
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
OUT = out_path("knowledge", f"{a.model}.json")
SHARD = lambda k, n: OUT.with_name(f"{a.model}_shard{k}of{n}.json")

if a.merge:
    fs = sorted(glob.glob(str(OUT.with_name(f"{a.model}_shard*.json"))))
    rows = sorted((r for f in fs for r in json.load(open(f))["rows"]), key=lambda r: r["kept"])
    json.dump(dict(meta=json.load(open(fs[0]))["meta"], rows=rows), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(rows)} configurations")
    raise SystemExit

import lm_eval
from lm_eval.models.huggingface import HFLM
t0 = time.time()
R = Runner(a.model)
lm = HFLM(pretrained=R.model, tokenizer=R.tok, batch_size=a.batch_size)
k, n = map(int, a.shard.split("/"))
res = dict(meta=dict(model=a.model, n_layers=R.n_layers, tasks=TASKS, limit=a.limit, num_fewshot=0), rows=[])
for c in (list(range(R.n_layers - 1)) + [None])[k - 1::n]:
    with hook_ctx(truncate(R, c)):
        out = lm_eval.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, limit=a.limit,
                                      log_samples=False, verbosity="ERROR")["results"]
    r = dict(cut=c, kept=R.n_layers if c is None else c + 1, acc={t: float(out[t]["acc,none"]) for t in TASKS})
    res["rows"].append(r)
    print(f"[{(time.time() - t0) / 60:5.1f}m] kept {r['kept']:2d}  " + " ".join(f"{t}:{r['acc'][t]:.3f}" for t in TASKS), flush=True)
    json.dump(res, open(SHARD(k, n), "w"), indent=1)
