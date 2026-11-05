"""Table 1, right: replace the Encode block. Conditions: intact, block removed, mixers zeroed (MLPs
kept), and a token-wise linear projection h_out = h_in + M(h_in) fitted by ridge regression on intact
runs. Metrics: accuracy on prompts with keys never seen in the fit, WikiText-2 perplexity.

    python -m experiments.encode_replace --model 9B
"""
import json, argparse, time, random
import numpy as np
import torch
from linquery.config import group_size, out_path
from linquery.runner import Runner, hook_ctx
from linquery.xfam import key_split, make_items, text_items
from linquery.stitch import InCap, OutCap, Replace, ZeroMix, Ridge

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="9B")
ap.add_argument("--n_fit", type=int, default=1500)
ap.add_argument("--n_text", type=int, default=200)
ap.add_argument("--n_test", type=int, default=200)
a = ap.parse_args()
OUT = out_path("encode", f"replace_{a.model}.json")
t0 = time.time()
Y = Runner(a.model)
ENC = [G for G in Y.groups(group_size(a.model)) if G][0]
rng = random.Random(0)
K_FIT, K_TEST = key_split()
ALL = make_items(rng, Y, None, K_FIT, a.n_fit, "chat") + text_items(Y, None, a.n_text)
test = make_items(rng, Y, None, K_TEST, a.n_test, "chat")
bk = {}
for it in test:
    bk.setdefault((len(it["y"]), str(it["yent"])), []).append(it)
TB = [v[i:i + 25] for v in bk.values() for i in range(0, len(v), 25)]
from datasets import load_dataset
wt = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
wids = Y.tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
WSEQ = torch.stack([wids[i * 512:(i + 1) * 512] for i in range(8)]).to(Y.device)


@torch.no_grad()
def evaluate(hooks):
    ok = n = 0
    with hook_ctx(list(hooks)):
        for bt in TB:
            ids = torch.tensor([it["y"] for it in bt], device=Y.device)
            lg = Y.model(ids, use_cache=False).logits[:, -1].float()
            aid = torch.tensor([it["aid"] for it in bt], device=lg.device)
            ok += (lg.argmax(-1) == aid).sum().item(); n += len(bt)
        nll = []
        for i in range(0, len(WSEQ), 4):
            x = WSEQ[i:i + 4]
            lg = Y.model(x, use_cache=False).logits[:, :-1].float()
            nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none"))
    return dict(acc=ok / n, ppl=float(torch.exp(torch.cat(nll).mean())))


X, T = [], []
with torch.no_grad():
    for it in ALL:
        ci, co = InCap(Y, ENC[0]), OutCap(Y, ENC[-1])
        with hook_ctx([ci, co]):
            Y.model(torch.tensor([it["y"]], device=Y.device), use_cache=False)
        X.append(ci.v[0].cpu()); T.append((co.v[0] - ci.v[0]).cpu())
proj = Ridge(torch.cat(X), torch.cat(T))
RB = lambda g: Replace(Y, ENC[0], ENC[-1], g)
res = dict(meta=dict(model=a.model, encode=ENC, n_fit=a.n_fit, n_text=a.n_text, n_test=len(test), proj_r2=proj.r2), conditions={})
C = res["conditions"]
C["intact"] = evaluate([])
C["removed"] = evaluate([RB(None)])
C["mixers zeroed"] = evaluate([ZeroMix(Y, L) for L in ENC])
C["linear projection"] = evaluate([RB(proj)])
for k, v in C.items():
    print(f"{k:18s} acc {v['acc']:.3f}  ppl {v['ppl']:.1f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"wrote {OUT} ({(time.time() - t0) / 60:.1f} min)", flush=True)
