"""Shared prompt material for the cross-family experiments.

Keys and values are single tokens (with and without a leading space) in the Qwen3.5 and
Granite4.0 tokenizers, so one dictionary text is a well-formed task for both families.
"""
import random
from .prompts import Builder, chat_wrap

KEYS = """apple banana cherry date elder fig grape lemon mango peach pear plum berry onion garlic ginger
pepper tomato carrot potato walnut almond lion tiger bear wolf fox deer horse sheep goat cow pig dog cat
mouse rabbit eagle hawk owl duck goose crow snake frog fish shark whale seal chair table desk lamp door
window bed sofa clock mirror bottle cup plate bowl spoon fork knife pen book paper phone camera radio
piano guitar drum violin river lake ocean mountain hill forest desert island valley beach cloud rain snow
storm wind moon star planet sun sky king queen prince doctor nurse teacher farmer pilot captain soldier
judge lawyer artist singer dancer writer train plane boat ship truck car bus bike rocket wagon bread
cheese butter milk honey sugar rice pasta soup cake pie cookie candy coffee tea juice wine beer water
shirt dress coat hat shoe sock glove scarf belt ring watch hammer nail brush rope chain wheel box bag
basket key lock bell flag map coin""".split()
VALUES = ["red", "blue", "green", "gold", "gray", "black", "white", "orange", "yellow", "amber", "rose",
          "lime", "azure", "steel", "ice", "fire", "stone", "wood", "iron", "silver", "pink", "purple",
          "brown", "glass", "sand", "salt", "ash", "ruby", "mint", "tan"]

def key_split():
    keys = [k for k in KEYS if k not in VALUES]
    random.Random(0).shuffle(keys)
    return keys[:len(keys) // 2], keys[len(keys) // 2:]


def render(wrap, pairs, q, style="chat"):
    """Returns (text, entry spans, segments). Segments are the (start, end) character ranges of
    the parts shared by every family: the user body and the assistant prefix."""
    b = Builder(); es = []
    b.add(wrap[0])
    s0 = len(b.s)
    if style == "list":
        b.add("Lookup table:\n")
        for k, v in pairs:
            e0, _ = b.add(f"* {k}={v}"); _, e1 = b.add("\n"); es.append((e0, e1))
        b.add(f"Which value does {q} map to?")
        ans = f"The entry {q} maps to **"
    elif style == "rev":
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, e1 = b.add(f" {v} is the value of {k}"); es.append((e0, e1))
            b.add("," if i < len(pairs) - 1 else ".")
        b.add(f"\nQuestion: What is the value of {q}?")
        ans = f"The value of {q} is **"
    else:
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, e1 = b.add(f" {k}={v}"); es.append((e0, e1))
            b.add("," if i < len(pairs) - 1 else ".")
        b.add(f"\nQuestion: What is the value of {q}?")
        ans = f"The value of {q} is **"
    s1 = len(b.s)
    b.add(wrap[1])
    a0, a1 = b.add(ans)
    return b.s, es, [(s0, s1), (a0, a1)]


def encode(tok, pairs, q, style="chat"):
    """input ids, entry token positions, and per-token canonical spans (segment, start, end)
    relative to the shared segments (None for wrapper tokens)."""
    s, es, segs = render(chat_wrap(tok), pairs, q, style)
    e = tok(s, return_offsets_mapping=True)
    ent = [[i for i, (x, y) in enumerate(e.offset_mapping) if x < b_ and y > a_] for a_, b_ in es]
    canon = []
    for x, y in e.offset_mapping:
        c = None
        for si, (a_, b_) in enumerate(segs):
            if a_ <= x and y <= b_ and y > x:
                c = (si, x - a_, y - a_)
        canon.append(c)
    return e.input_ids, ent, canon


def align(canon_a, canon_b):
    """Pairs of positions (i in a, j in b) whose tokens cover the same shared characters."""
    ib = {c: j for j, c in enumerate(canon_b) if c is not None}
    return [(i, ib[c]) for i, c in enumerate(canon_a) if c is not None and c in ib]


def make_items(rng, Y, D, keyset, n, style):
    """Dictionaries of 8 pairs on `keyset`, one queried key; tokenised for receiver Y and donor D."""
    out = []
    while len(out) < n:
        ks = rng.sample(keyset, 8); vs = rng.sample(VALUES, 8); ia = rng.randrange(8)
        pairs = list(zip(ks, vs))
        it = dict(pairs=pairs, ia=ia)
        it["y"], it["yent"], cy = encode(Y.tok, pairs, ks[ia], style)
        if D is not None:
            it["d"], _, cd = encode(D.tok, pairs, ks[ia], style)
            it["al"] = align(cy, cd)
        aid = Y.tok(vs[ia]).input_ids
        if len(aid) == 1:
            it["aid"] = aid[0]
            it["cand"] = [Y.tok(v).input_ids[0] for v in vs]
            out.append(it)
    return out


def text_items(Y, D, n):
    """WikiText chunks added to the map fits."""
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    out = []
    for t in [t for t in d["text"] if len(t) > 700][:n]:
        t = t[:600]
        ey = Y.tok(t, return_offsets_mapping=True)
        it = dict(y=ey.input_ids)
        if D is not None:
            ed = D.tok(t, return_offsets_mapping=True)
            cy = [(0, x, y) if y > x else None for x, y in ey.offset_mapping]
            cd = [(0, x, y) if y > x else None for x, y in ed.offset_mapping]
            it.update(d=ed.input_ids, al=align(cy, cd))
        out.append(it)
    return out
