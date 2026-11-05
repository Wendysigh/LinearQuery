"""Table 3 and Appendix C, removal: zero the mixer outputs of a whole set of linear layers jointly.

    python -m experiments.remove_blocks --model 27B --format chat8        (the model's Address blocks)
"""
import json, argparse, time, torch
import numpy as np
from linquery.config import address_layers, out_path
from linquery.runner import Runner, hook_ctx, Zero
from linquery.prompts import make_items, batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--format", default="chat8")
ap.add_argument("--max_bs", type=int, default=25)
a = ap.parse_args()
t0 = time.time()
R = Runner(a.model)
B = batches(R.tok, make_items(a.format, R.tok), R.device, a.max_bs)


@torch.no_grad()
def evaluate(layers):
    ok = tot = 0; gaps = []
    for b in B:
        ds = []
        for ids, want in ((b.clean_ids, b.aid), (b.corr_ids, b.cid)):
            with hook_ctx([Zero(R, L) for L in layers]):
                lg = R.model(ids, use_cache=False).logits[:, -1]
            ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
            ds.append(b.D(lg))
        gaps += (ds[0] - ds[1]).tolist()
    return dict(acc=ok / tot, gap=float(np.mean(gaps)))


Ls = [L for L in address_layers(a.model) if L in R.gdn_layers]    # softmax layers inside a range stay
res = dict(meta=dict(model=a.model, format=a.format, n=sum(len(b.items) for b in B), layers=Ls),
           baseline=evaluate([]), removed=evaluate(Ls))
print(f"[{a.model} {a.format}] acc {res['baseline']['acc']:.3f} -> {res['removed']['acc']:.3f} "
      f"gap {res['baseline']['gap']:+.2f} -> {res['removed']['gap']:+.2f}", flush=True)
json.dump(res, open(out_path("removal", f"{a.model}_{a.format}.json"), "w"), indent=2)
print(f"done ({(time.time() - t0) / 60:.1f} min)", flush=True)
