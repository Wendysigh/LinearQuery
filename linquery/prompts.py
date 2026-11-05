"""Prompts of the retrieval task. Clean/corrupted items (accuracy, removal) and donor/host pairs
(transplant) have identical token length within a pair; token positions are recovered from
character spans via the fast tokenizer's offset mapping, so any template works.

Formats: chat{N} (N pairs), list8 (bulleted lookup table, rephrased question), rev8 (value before
key), long{L} (chat8 with ~L tokens of WikiText between the dictionary and the question).
"""
import random
from dataclasses import dataclass, field
import torch

# Candidate keys and values. Every word is a single token in the rendered context.
KEY_WORDS = ["apple", "banana", "cherry", "date", "elder", "fig", "grape", "lemon", "mango",
             "peach", "pear", "plum", "berry", "onion", "garlic", "ginger", "pepper", "tomato",
             "carrot", "potato", "walnut", "almond"]
VALUE_WORDS = ["red", "blue", "green", "gold", "gray", "black", "white", "orange", "yellow",
               "amber", "rose", "lime", "olive", "azure", "steel", "ice", "fire", "stone",
               "wood", "iron"]

CHAT_PRE = "<|im_start|>user\n"
CHAT_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def chat_wrap(tok):
    """(prefix, suffix) that put a user turn and open the assistant turn, for any model.

    Qwen3.5 uses the strings above (empty think block, so the answer is the next token); other
    families are derived from the tokenizer's own chat template."""
    tpl = getattr(tok, "chat_template", None) or ""
    if "<|im_start|>" in tpl:
        return CHAT_PRE, CHAT_POST
    s = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                add_generation_prompt=True)
    pre, post = s.split("\x00", 1)
    return pre, post


@dataclass
class Item:
    clean: str
    corr: str
    answer: str                  # surface form of the next token (clean)
    answer_corr: str
    target_idx: int
    distract_idx: int
    value_spans: list            # per entry: (start, end) char span of its value/letter token
    entry_spans: list            # per entry: (start, end) char span of the whole entry
    meta: dict = field(default_factory=dict)


class Builder:
    """Append text pieces while recording character spans."""

    def __init__(self): self.s = ""
    def add(self, t):
        a = len(self.s); self.s += t
        return a, len(self.s)


def _dict_item(pairs, pairs_c, ti, di, style, filler="", wrap=(CHAT_PRE, CHAT_POST)):
    def render(ps):
        b = Builder(); vs, es = [], []
        b.add(wrap[0])
        q = ps[ti][0]
        if style == "list":
            b.add("Lookup table:\n")
            for k, v in ps:
                e0, _ = b.add(f"* {k}="); v0, v1 = b.add(v); _, e1 = b.add("\n")
                vs.append((v0, v1)); es.append((e0, e1))
            b.add(filler)
            b.add(f"Which value does {q} map to?" + wrap[1] + f"The entry {q} maps to **")
        elif style == "rev":
            b.add("Dictionary:")
            for i, (k, v) in enumerate(ps):
                e0, _ = b.add(" "); v0, v1 = b.add(v); _, e1 = b.add(f" is the value of {k}")
                b.add("," if i < len(ps) - 1 else ".")
                vs.append((v0, v1)); es.append((e0, e1))
            b.add(filler)
            b.add(f"\nQuestion: What is the value of {q}?" + wrap[1] + f"The value of {q} is **")
        else:
            b.add("Dictionary:")
            for i, (k, v) in enumerate(ps):
                e0, _ = b.add(f" {k}="); v0, v1 = b.add(v); _, e1 = b.add("," if i < len(ps) - 1 else ".")
                vs.append((v0, v1)); es.append((e0, e1))
            b.add(filler)
            b.add(f"\nQuestion: What is the value of {q}?" + wrap[1] + f"The value of {q} is **")
        return b.s, vs, es
    c, vs, es = render(pairs)
    x, _, _ = render(pairs_c)
    return Item(c, x, pairs[ti][1], pairs_c[ti][1], ti, di, vs, es)


def filler_text(tok, n_tokens):
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
    ids = tok("\n\n".join(t for t in d["text"][200:] if t.strip())).input_ids[:n_tokens]
    return "\n" + tok.decode(ids).strip() + "\n"


def make_items(variant, tok, n_per_cfg=25):
    wrap = chat_wrap(tok)
    rng = random.Random(0)
    style, n_pairs, filler = "chat", 8, ""
    if variant.startswith("chat"):
        n_pairs = int(variant[4:])
    elif variant == "list8":
        style = "list"
    elif variant == "rev8":
        style = "rev"
    elif variant.startswith("long"):
        filler = filler_text(tok, int(variant[4:]))
    cfgs = {4: [(0, 3), (3, 0), (1, 2), (2, 1)], 16: [(3, 12), (12, 3), (6, 9), (9, 6)]}.get(
        n_pairs, [(2, 5), (5, 2), (1, 6), (6, 1)])
    vals = [v for v in VALUE_WORDS if len(tok(v).input_ids) == 1 and len(tok(" " + v).input_ids) == 1]
    out = []
    for ti, di in cfgs:
        for _ in range(n_per_cfg):
            keys = rng.sample(KEY_WORDS, n_pairs)
            vs = rng.sample(vals, n_pairs) if n_pairs <= len(vals) else None
            pairs = list(zip(keys, vs))
            pc = list(pairs)
            pc[ti] = (keys[ti], vs[di]); pc[di] = (keys[di], vs[ti])
            out.append(_dict_item(pairs, pc, ti, di, style, filler, wrap))
    return out


def positions(tok, it):
    """Token positions from the clean prompt's char spans. Returns None unless clean and corrupted
    prompts have equal token length and differ only inside the target and distractor entries,
    every value/letter is a single token, and both answers are single tokens."""
    ec = tok(it.clean, return_offsets_mapping=True)
    ex = tok(it.corr, return_offsets_mapping=True)
    if len(ec.input_ids) != len(ex.input_ids):
        return None
    off = ec.offset_mapping

    def toks(span):
        a, b = span
        return [i for i, (s, e) in enumerate(off) if s < b and e > a]
    vpos, epos = [], []
    for vs, es in zip(it.value_spans, it.entry_spans):
        v = toks(vs)
        if len(v) != 1:
            return None
        vpos.append(v[0]); epos.append(toks(es))
    diff = {i for i, (x, y) in enumerate(zip(ec.input_ids, ex.input_ids)) if x != y}
    if not diff or not diff <= set(epos[it.target_idx]) | set(epos[it.distract_idx]):
        return None
    ans = tok(it.answer).input_ids, tok(it.answer_corr).input_ids
    if len(ans[0]) != 1 or len(ans[1]) != 1:
        return None
    return dict(n=len(ec.input_ids), final=len(ec.input_ids) - 1, value_pos=vpos, entry_pos=epos,
                ids=ec.input_ids, ids_c=ex.input_ids, aid=ans[0][0], cid=ans[1][0])


class Batch:
    """Items sharing length and all positions, run together."""

    def __init__(self, items, poss, device):
        p = poss[0]
        self.items, self.pos = items, p
        self.ti = items[0].target_idx
        self.clean_ids = torch.tensor([q["ids"] for q in poss], device=device)
        self.corr_ids = torch.tensor([q["ids_c"] for q in poss], device=device)
        self.aid = torch.tensor([q["aid"] for q in poss], device=device)
        self.cid = torch.tensor([q["cid"] for q in poss], device=device)

    def D(self, logits):
        lg = logits.float()
        return (lg.gather(1, self.aid[:, None]) - lg.gather(1, self.cid[:, None])).squeeze(1)


def batches(tok, items, device, max_bs=25):
    buckets = {}
    for it in items:
        p = positions(tok, it)
        if p is None:
            continue
        key = (p["n"], tuple(p["value_pos"]), tuple(map(tuple, p["entry_pos"])), it.target_idx)
        buckets.setdefault(key, []).append((it, p))
    out = []
    for v in buckets.values():
        for i in range(0, len(v), max_bs):
            ch = v[i:i + max_bs]
            out.append(Batch([c[0] for c in ch], [c[1] for c in ch], device))
    return out


# ---------------------------------------------------------------- donor/host pairs
def _pair_render(wrap, pairs, q, style):
    """Prompt asking for key `q`: text, value spans, entry spans, start of the question."""
    b = Builder(); vs, es = [], []
    b.add(wrap[0])
    if style == "list":
        b.add("Lookup table:\n")
        for k, v in pairs:
            e0, _ = b.add(f"* {k}="); v0, v1 = b.add(v); _, e1 = b.add("\n")
            vs.append((v0, v1)); es.append((e0, e1))
        qs0, _ = b.add(f"Which value does {q} map to?")
        b.add(wrap[1] + f"The entry {q} maps to **")
    elif style == "rev":
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, _ = b.add(" "); v0, v1 = b.add(v); _, e1 = b.add(f" is the value of {k}")
            b.add("," if i < len(pairs) - 1 else ".")
            vs.append((v0, v1)); es.append((e0, e1))
        qs0, _ = b.add(f"\nQuestion: What is the value of {q}?")
        b.add(wrap[1] + f"The value of {q} is **")
    else:
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, _ = b.add(f" {k}="); v0, v1 = b.add(v)
            _, e1 = b.add("," if i < len(pairs) - 1 else ".")
            vs.append((v0, v1)); es.append((e0, e1))
        qs0, _ = b.add(f"\nQuestion: What is the value of {q}?")
        b.add(wrap[1] + f"The value of {q} is **")
    return b.s, vs, es, qs0


def pair_batches(tok, variant, device, n_pairs=8, per_cfg=25):
    """Donor run A and host run B share one dictionary and differ only in the queried key, so they
    have identical token length and entry positions. Batches share all positions."""
    def spans_to_pos(off, span):
        return [i for i, (s, e) in enumerate(off) if s < span[1] and e > span[0]]

    wrap = chat_wrap(tok)
    style = {"list8": "list", "rev8": "rev"}.get(variant, "chat")
    rng = random.Random(0)
    vals = [v for v in VALUE_WORDS if len(tok(v).input_ids) == 1 and len(tok(" " + v).input_ids) == 1]
    keys = [k for k in KEY_WORDS if len(tok(" " + k).input_ids) == 1]
    items = []
    for ia, ib in [(2, 5), (5, 2), (1, 6), (6, 1)]:          # (A index, B index) in the dictionary
        for _ in range(per_cfg):
            ks = rng.sample(keys, n_pairs); vs_ = rng.sample(vals, n_pairs)
            pairs = list(zip(ks, vs_))
            ta, _, _, _ = _pair_render(wrap, pairs, ks[ia], style)
            tb, _, espans, qs0 = _pair_render(wrap, pairs, ks[ib], style)
            ea = tok(ta, return_offsets_mapping=True); eb = tok(tb, return_offsets_mapping=True)
            if len(ea.input_ids) != len(eb.input_ids):
                continue
            aid = tok(vs_[ia]).input_ids; bid = tok(vs_[ib]).input_ids   # "is **lime": no space
            if len(aid) != 1 or len(bid) != 1:
                continue
            off = eb.offset_mapping
            items.append(dict(ids_a=ea.input_ids, ids_b=eb.input_ids, ia=ia, ib=ib,
                              aid=aid[0], bid=bid[0], n=len(eb.input_ids),
                              entry=[spans_to_pos(off, s) for s in espans],
                              qstart=min(spans_to_pos(off, (qs0, qs0 + 1)))))
    buckets = {}
    for it in items:
        k = (it["n"], tuple(map(tuple, it["entry"])), it["qstart"], it["ia"], it["ib"])
        buckets.setdefault(k, []).append(it)
    B = []
    for v in buckets.values():
        B.append(dict(ia=v[0]["ia"], ib=v[0]["ib"], n=v[0]["n"], entry=v[0]["entry"],
                      qstart=v[0]["qstart"],
                      ids_a=torch.tensor([x["ids_a"] for x in v], device=device),
                      ids_b=torch.tensor([x["ids_b"] for x in v], device=device),
                      aid=torch.tensor([x["aid"] for x in v], device=device),
                      bid=torch.tensor([x["bid"] for x in v], device=device)))
    return B
