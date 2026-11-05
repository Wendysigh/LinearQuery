"""Fig. 4b, c: what carries the address, and does the Address block need its upstream blocks?

Panel b, each measured as ansA (the host answers the donor's value):
  out_final       Address mixer output transplanted at the final token
  state_G{4,3,0}  that block's recurrent states after tokens 0..n-2, donor's patched into the host
  input_final     the residual input of the first Address layer at the final token
Panel c: donor run with an upstream block zeroed at every position (G1, G2, G3, or all three); its
Address output at the final token is transplanted into the intact host.

    python -m experiments.construction --format chat8
"""
import json, argparse, time, torch
import numpy as np
from linquery.config import out_path
from linquery.runner import Runner, hook_ctx, Capture, Transplant, Zero, InPatch
from linquery.prompts import pair_batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="9B")
ap.add_argument("--format", default="chat8")
ap.add_argument("--address", type=int, default=4)
a = ap.parse_args()
OUT = out_path("construction", f"{a.model}_{a.format}.json")
t0 = time.time()
R = Runner(a.model)
GROUPS = R.groups(3)
ADDR = GROUPS[a.address]
B = pair_batches(R.tok, a.format, R.device)
N = sum(len(b["aid"]) for b in B)
print(f"  {N} prompt pairs in {len(B)} batches, {B[0]['n']} tokens", flush=True)


class LayerInCapture:
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(mod, args, kwargs):
            t = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            self.value = t.detach().clone()
        return R.layers[self.L].register_forward_pre_hook(fn, with_kwargs=True)


def score(lg, b):
    lg = lg.float(); am = lg.argmax(-1)
    return dict(ansA=float((am == b["aid"]).float().mean()), ansB=float((am == b["bid"]).float().mean()),
                dAB=float((lg.gather(1, b["aid"][:, None]) - lg.gather(1, b["bid"][:, None])).mean()))


@torch.no_grad()
def final_logits(ids, hooks=()):
    with hook_ctx(hooks):
        return R.model(ids, use_cache=False).logits[:, -1]


@torch.no_grad()
def donor_outputs(ids, zero=()):
    caps = [Capture(R, L) for L in ADDR]
    with hook_ctx([Zero(R, L) for L in zero] + caps):
        R.model(ids, use_cache=False)
    return caps


conds = {}
for b in B:
    n = b["n"]; fin = [n - 1]
    conds.setdefault("host_intact", []).append(score(final_logits(b["ids_b"]), b))
    caps = donor_outputs(b["ids_a"])
    lg = final_logits(b["ids_b"], [Transplant(R, L, fin, c.value) for L, c in zip(ADDR, caps)])
    conds.setdefault("out_final", []).append(score(lg, b))
    donor_states = R.recurrent_states(b["ids_a"], n - 2)
    for gi in (a.address, a.address - 1, 0):
        lg = R.split_forward(b["ids_b"], n - 2, state_patch={L: donor_states[L] for L in GROUPS[gi]})
        conds.setdefault(f"state_G{gi}", []).append(score(lg, b))
    ci = LayerInCapture(ADDR[0])
    with torch.no_grad(), hook_ctx([ci]):
        R.model(b["ids_a"], use_cache=False)
    lg = final_logits(b["ids_b"], [InPatch(R.layers[ADDR[0]], fin, ci.value)])
    conds.setdefault("input_final", []).append(score(lg, b))
    for name, gs in [("noG1", [1]), ("noG2", [2]), ("noG3", [3]), ("noG1-3", [1, 2, 3])]:
        caps = donor_outputs(b["ids_a"], [L for g in gs for L in GROUPS[g]])
        lg = final_logits(b["ids_b"], [Transplant(R, L, fin, c.value) for L, c in zip(ADDR, caps)])
        conds.setdefault(f"up_{name}", []).append(score(lg, b))

res = dict(meta=dict(model=a.model, format=a.format, n=N, address=ADDR), rows={})
for k, v in conds.items():
    res["rows"][k] = {m: float(np.mean([r[m] for r in v])) for m in v[0]}
    print(f"{a.model} {a.format} {k:14s} ansA {res['rows'][k]['ansA']:.2f}  ansB {res['rows'][k]['ansB']:.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT} ({(time.time() - t0) / 60:.1f} min)", flush=True)
