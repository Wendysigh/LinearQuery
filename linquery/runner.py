"""Model loading and split execution for GDN / Mamba-2 hybrids. Weights are frozen throughout."""
import contextlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from .config import model_path


class Runner:
    def __init__(self, tag, device="cuda:0"):
        path = model_path(tag)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        cfg = self.model.config.get_text_config()
        cfg._attn_implementation = "eager"      # attention weights are recorded
        self.cfg = cfg
        self.layer_types = list(cfg.layer_types)
        # "linear_attention" (Qwen3.5 GDN) / "mamba" (Granite4.0) vs the softmax layers
        self.gdn_layers = [i for i, t in enumerate(self.layer_types)
                           if t in ("linear_attention", "mamba")]
        self.attn_layers = [i for i, t in enumerate(self.layer_types)
                            if t in ("full_attention", "attention")]
        self.n_layers = cfg.num_hidden_layers

    @property
    def head_dim(self):
        return getattr(self.cfg, "head_dim", None) or (
            self.cfg.hidden_size // self.cfg.num_attention_heads)

    @property
    def q_gated(self):
        """True when q_proj emits [query; gate] per head (Qwen3.5) rather than the query only."""
        q = self.mixer(self.attn_layers[0]).q_proj
        return q.out_features == 2 * self.cfg.num_attention_heads * self.head_dim

    def groups(self, size=3):
        """The linear layers feeding each softmax layer: the `size` immediately before it,
        or all of them since the previous softmax layer when size <= 0."""
        out, prev = [], -1
        for s in self.attn_layers:
            lo = prev + 1 if size <= 0 else s - size
            out.append([L for L in range(max(lo, 0), s) if L in self.gdn_layers])
            prev = s
        return out

    @property
    def layers(self):
        return self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers

    def mixer(self, i):
        """The layer's token-mixing module, whatever the family calls it."""
        L = self.layers[i]
        # attention layers of some families keep a `mamba` attribute set to None, so the
        # layer's own type decides which names are eligible
        names = (("linear_attn", "mamba") if self.layer_types[i] in ("linear_attention", "mamba")
                 else ("self_attn",))
        for name in names:
            m = getattr(L, name, None)
            if m is not None:
                return m
        raise AttributeError(f"no {names} mixer on layer {i}: {[n for n, _ in L.named_children()]}")

    @torch.no_grad()
    def run_prefix(self, ids, upto, hooks=()):
        """Run tokens [0, upto] (inclusive) and return a fresh cache."""
        cache = DynamicCache(config=self.model.config.get_text_config())
        with hook_ctx(hooks):
            self.model(ids[:, : upto + 1], past_key_values=cache, use_cache=True)
        return cache

    @torch.no_grad()
    def run_suffix(self, ids, upto, cache, hooks=()):
        """Run tokens [upto+1, end] on top of `cache`; returns final-position logits."""
        n = ids.shape[1]
        pos = torch.arange(upto + 1, n, device=ids.device).view(1, -1)
        with hook_ctx(hooks):
            out = self.model(ids[:, upto + 1:], past_key_values=cache, use_cache=True, position_ids=pos)
        return out.logits[:, -1, :]

    @torch.no_grad()
    def split_forward(self, ids, upto, state_patch=None):
        """Run the prefix, optionally patch recurrent states {layer: tensor}, then run the suffix."""
        cache = self.run_prefix(ids, upto)
        if state_patch:
            for li, st in state_patch.items():
                cache.layers[li].recurrent_states[0] = st.clone().to(
                    cache.layers[li].recurrent_states[0].dtype)
        return self.run_suffix(ids, upto, cache)

    @torch.no_grad()
    def recurrent_states(self, ids, upto):
        """Recurrent states after consuming tokens [0, upto]; {layer: tensor (clone)}."""
        cache = self.run_prefix(ids, upto)
        return {li: cache.layers[li].recurrent_states[0].detach().clone() for li in self.gdn_layers}


@contextlib.contextmanager
def hook_ctx(hooks):
    handles = []
    try:
        for h in hooks:
            handles.append(h.register())
        yield
    finally:
        for h in handles:
            h.remove()


def _t(out):
    return out[0] if isinstance(out, tuple) else out


def _put(out, t):
    return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t


class Zero:
    """Zero a token mixer's output (linear or softmax: attention layers return a tuple)."""
    def __init__(self, R, L): self.R, self.L = R, L
    def register(self):
        return self.R.mixer(self.L).register_forward_hook(
            lambda m, a, o: _put(o, torch.zeros_like(_t(o))))


class Capture:
    """Record a mixer's output."""
    def __init__(self, R, L): self.R, self.L, self.value = R, L, None
    def register(self):
        def fn(m, args, out):
            self.value = _t(out).detach().clone()
        return self.R.mixer(self.L).register_forward_hook(fn)


class Transplant:
    """Overwrite a mixer's output at `positions` with the donor run's output."""
    def __init__(self, R, L, positions, donor): self.R, self.L, self.pos, self.donor = R, L, positions, donor
    def register(self):
        def fn(m, args, out):
            t = _t(out).clone()
            t[:, self.pos] = self.donor[:, self.pos].to(t.dtype)
            return _put(out, t)
        return self.R.mixer(self.L).register_forward_hook(fn)


class InPatch:
    """Overwrite a module's `hidden_states` input at selected positions with donor values."""
    def __init__(self, module, positions, donor):
        self.module, self.positions, self.donor = module, positions, donor

    def register(self):
        def fn(mod, args, kwargs):
            t = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).clone()
            for p in self.positions:
                t[:, p] = self.donor[:, p].to(t.dtype)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = t
                return args, kwargs
            return (t,) + tuple(args[1:]), kwargs
        return self.module.register_forward_pre_hook(fn, with_kwargs=True)
