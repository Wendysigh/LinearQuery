"""Block stitching: residual captures, block replacement, and closed-form ridge maps."""
import numpy as np
import torch
from .runner import hook_ctx


class InCap:
    """Residual stream entering layer L (all positions)."""
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(mod, args, kwargs):
            self.v = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).detach().float()
        return self.M.layers[self.L].register_forward_pre_hook(fn, with_kwargs=True)


class OutCap:
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(mod, args, out):
            self.v = (out[0] if isinstance(out, tuple) else out).detach().float()
        return self.M.layers[self.L].register_forward_hook(fn)


class _Handles:
    def __init__(self, hs): self.hs = hs
    def remove(self):
        for x in self.hs:
            x.remove()


class Replace:
    """Skip layers lo..hi-1 and make layer hi output h_in + g(h_in) (g=None: h_out = h_in)."""
    def __init__(self, M, lo, hi, g): self.M, self.lo, self.hi, self.g = M, lo, hi, g
    def register(self):
        hs = []
        for L in range(self.lo, self.hi + 1):
            def fn(mod, args, kwargs, out, L=L):
                h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                new = h if (L < self.hi or self.g is None) else h + self.g(h.float()).to(h.dtype)
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            hs.append(self.M.layers[L].register_forward_hook(fn, with_kwargs=True))
        return _Handles(hs)


class Cut:
    """Skip every layer after `cut` (the hidden state passes through to the final norm)."""
    def __init__(self, M, cut): self.M, self.cut = M, cut
    def register(self):
        hs = []
        for L in range(self.cut + 1, self.M.n_layers):
            def fn(mod, args, kwargs, out):
                h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hs.append(self.M.layers[L].register_forward_hook(fn, with_kwargs=True))
        return _Handles(hs)


class ZeroMix:
    def __init__(self, M, L): self.M, self.L = M, L
    def register(self):
        return self.M.mixer(self.L).register_forward_hook(
            lambda m, a_, o: (torch.zeros_like(o[0]),) + tuple(o[1:]) if isinstance(o, tuple)
            else torch.zeros_like(o))


@torch.no_grad()
def run_block(M, block, h, nomix=False):
    """Decoder layers `block` of model M applied to hidden states h [1, T, H]."""
    hooks = [ZeroMix(M, L) for L in block] if nomix else []
    with hook_ctx(hooks):
        x = h.to(next(M.model.parameters()).dtype)
        for L in block:
            o = M.layers[L](x, position_embeddings=None)
            x = o[0] if isinstance(o, tuple) else o
    return x.float()


class Ridge:
    """Centred ridge regression, lambda chosen on a random held-out fifth (<= 4000 rows)."""
    def __init__(self, X, T):
        n = len(X); perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        n_val = min(4000, n // 5)
        vi, ti = perm[:n_val], perm[n_val:]
        X = X.double().cuda(); T = T.double().cuda()
        self.mx, self.mt = X[ti].mean(0), T[ti].mean(0)
        Xc, Tc = X[ti] - self.mx, T[ti] - self.mt
        ev, V = torch.linalg.eigh(Xc.T @ Xc)
        VC = V.T @ (Xc.T @ Tc)
        Xv, Tv = X[vi] - self.mx, T[vi] - self.mt
        best = None
        for p_ in np.arange(-6, 2.5, 0.5):
            lam = 10 ** p_ * float(ev[-1])
            W = V @ (VC / (ev + lam)[:, None])
            err = float(((Xv @ W - Tv) ** 2).sum() / (Tv ** 2).sum())
            if best is None or err < best[0]:
                best = (err, W)
        self.W, self.r2 = best[1], 1 - best[0]

    def __call__(self, X):
        return ((X.double() - self.mx) @ self.W + self.mt).float()
