"""Truncation hooks and the evaluators of the truncation curve."""
import numpy as np
import torch
from .runner import hook_ctx, _put
from .prompts import make_items, batches


class SkipLayer:
    def __init__(self, R, L): self.R, self.L = R, L
    def register(self):
        def fn(mod, args, kwargs, output):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            return _put(output, h)
        return self.R.layers[self.L].register_forward_hook(fn, with_kwargs=True)


def truncate(R, cut):
    """Keep layers 0..cut; every later layer (mixer and MLP) is skipped, so the hidden state goes
    straight to the final norm and the LM head. cut=None: full depth."""
    last = R.n_layers - 1 if cut is None else cut
    return [SkipLayer(R, L) for L in range(last + 1, R.n_layers)]


class DictSuite:
    """acc: full-vocabulary argmax is the answer. cacc: the answer has the highest logit among the
    dictionary's values (chance 1/8)."""

    def __init__(self, R, variants=("chat8", "chat16", "list8", "rev8", "long512"), max_bs=25):
        self.R = R
        self.B = {}
        for v in variants:
            self.B[v] = batches(R.tok, make_items(v, R.tok), R.device, max_bs)
            for b in self.B[v]:          # candidates = the bare value words (the answer's form)
                b.cand = torch.tensor([[R.tok(it.clean[s0:s1]).input_ids[0] for s0, s1 in it.value_spans]
                                       for it in b.items], device=R.device)

    @torch.no_grad()
    def run(self, hooks=()):
        out = {}
        with hook_ctx(hooks):
            for v, B in self.B.items():
                ok = cok = tot = 0; gaps = []
                for b in B:
                    ds = []
                    for ids, want in ((b.clean_ids, b.aid), (b.corr_ids, b.cid)):
                        lg = self.R.model(ids, use_cache=False).logits[:, -1].float()
                        ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
                        cand = b.cand
                        pick = cand.gather(1, lg.gather(1, cand).argmax(-1, keepdim=True))[:, 0]
                        cok += (pick == want).sum().item()
                        ds.append(b.D(lg))
                    gaps += (ds[0] - ds[1]).tolist()
                out[v] = dict(acc=ok / tot, cacc=cok / tot, gap=float(np.mean(gaps)))
        return out


class WikiPPL:
    """WikiText-2 test perplexity."""
    def __init__(self, R, n_seq=16, seq_len=512):
        from datasets import load_dataset
        d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
        ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
        self.R = R
        self.x = torch.stack([ids[i * seq_len:(i + 1) * seq_len] for i in range(n_seq)]).to(R.device)

    @torch.no_grad()
    def run(self, hooks=(), bs=4):
        nll = []
        with hook_ctx(hooks):
            for i in range(0, len(self.x), bs):
                x = self.x[i:i + bs]
                lg = self.R.model(x, use_cache=False).logits[:, :-1].float()
                nll.append(torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none"))
        return float(torch.exp(torch.cat(nll).mean()))
