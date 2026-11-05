"""Tables and in-text numbers of the paper, from results/ (no GPU needed).

    python make_tables.py          print every table
    python make_tables.py --check  compare with the values printed in the paper (paper_values.json)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
V = ("chat8", "list8", "rev8")
BLOCK = {"0.8B": "G3", "2B": "G3", "4B": "G4", "9B": "G4"}


def load(p):
    f = RES / p
    return json.load(open(f)) if f.exists() else None


def final(path, name):
    """ansA at the final position when block set `name` is transplanted."""
    d = load(path)
    if d is None:
        return None
    for r in d["transplant"]:
        if r["positions"] == "final" and r["name"] == name:
            return r["ansA"]
    return None


def mean(xs):
    return None if any(x is None for x in xs) else float(np.mean(xs))


def get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d[k]
    return d


def table1(o):
    sw = load("encode/swap_9B.json")
    for row, key in (("encode", "encode"), ("embedding", "embedding"), ("no_patch", "no patch")):
        for s in ("value", "key"):
            o[f"t1.{row}.{s}"] = get(sw, "swaps", s, key, "success")
    for s in ("value", "key"):
        others = [v["success"] for k, v in sw["swaps"][s].items() if k.startswith("G")] if sw else None
        o[f"t1.other_blocks.{s}"] = max(others) if others else None
    rp = load("encode/replace_9B.json")
    for row, key in (("intact", "intact"), ("linear", "linear projection"), ("zeroed", "mixers zeroed"), ("removed", "removed")):
        o[f"t1.{row}.acc"] = get(rp, "conditions", key, "acc")
        o[f"t1.{row}.ppl"] = get(rp, "conditions", key, "ppl")


def table2(o):
    P = [("0.8B>G1B", "0.8B", "G1B", "", "G3"), ("G1B>0.8B", "G1B", "0.8B", "", "G2"),
         ("9B>G7B", "9B", "G7B", "_G2G3", "G4"), ("G7B>9B", "G7B", "9B", "_G2G3", "seq")]
    for name, d, r, sfx, addr in P:
        t = load(f"transfer/output_{d}_to_{r}_chat8{sfx}.json")
        o[f"t2.{name}.cross"] = get(t, "conditions", "src=G2+G3" if name == "G7B>9B" else "xfam", "ansA")
        o[f"t2.{name}.same"] = get(t, "conditions", "in-family", "ansA")
        b = load(f"transfer/block_{d}_into_{r}_chat8.json")
        cut = name == "G7B>9B"
        sec, m = ("cut", "cacc") if cut else ("full", "acc")
        units = list(get(b, "full") or {})
        don = next((u for u in units if ("seq" in u if cut else "(Address)" in u) and "zeroed" not in u), None)
        o[f"t2.{name}.removed"] = get(b, sec, "removed", m)
        o[f"t2.{name}.linear"] = get(b, sec, "per-token linear map", m)
        o[f"t2.{name}.zeroed"] = get(b, sec, f"{don}, mixers zeroed", m) if don else None
        o[f"t2.{name}.donor"] = get(b, sec, don, m) if don else None
        if cut:
            o[f"t2.{name}.intact"] = get(b, sec, "intact", m)
            o[f"t2.{name}.single_G2"] = get(b, sec, "donor G2 (Address)", m)
            o[f"t2.{name}.single_G3"] = get(b, sec, "donor G3", m)
    o["t2.9B>G7B.G2_only"] = get(load("transfer/output_9B_to_G7B_chat8_G2.json"), "conditions", "xfam", "ansA")
    xs = [get(load(f"transfer/output_{d}_to_{r}_{f}.json"), "conditions", "xfam", "ansA")
          for d, r in (("0.8B", "G1B"), ("G1B", "0.8B")) for f in V]
    o["text.small_pair_formats.min"] = min(xs) if None not in xs else None
    o["text.small_pair_formats.max"] = max(xs) if None not in xs else None


def table3(o):
    for t in ("0.8B", "2B", "4B", "9B", "27B", "35B"):
        o[f"t3.{t}.removal"] = mean([get(load(f"removal/{t}_{f}.json"), "removed", "acc") for f in V])
    for t in BLOCK:
        o[f"t3.{t}.transplant"] = mean([final(f"transplant/{t}_{f}.json", BLOCK[t]) for f in V])
    o["t3.27B.transplant"] = mean([final(f"transplant/27B_{f}_union.json", "G8+G9+G10+G11+G12") for f in V])
    o["t3.35B.transplant"] = mean([final(f"transplant/35B_{f}_union.json", "G6+G7+G8") for f in V])


def single_layer(o):
    for t, b, g in (("0.8B", "G3", 3), ("9B", "G4", 4), ("G1B", "G2", 2)):
        tr = load(f"transplant/{t}_chat8.json")
        ly = load(f"transplant/{t}_chat8_layers.json")
        sc = load(f"scan/{t}_chat8.json")
        rows = {r["name"]: r for r in tr["transplant"] if r["positions"] == "final"} if tr else {}
        o[f"sl.{t}.block"] = get(rows, b, "ansA"); o[f"sl.{t}.block_att"] = get(rows, b, "attA")
        fin = [r for r in ly["transplant"] if r["positions"] == "final"] if ly else []
        best = max(fin, key=lambda r: (r["ansA"], r["attA"])) if fin else None
        o[f"sl.{t}.layer"] = get(best, "ansA"); o[f"sl.{t}.layer_att"] = get(best, "attA")
        o[f"sl.{t}.remove_block"] = get(sc, "groups", g, "acc")
        Ls = get(sc, "groups", g, "layers")
        o[f"sl.{t}.remove_layer"] = min(r["acc"] for r in sc["layers"] if r["layer"] in Ls) if sc else None
        o[f"sl.{t}.remove_L0"] = get(sc, "layers", 0, "acc")


def models(o):
    for t, g in (("0.8B", 3), ("9B", 4), ("G1B", 2), ("G7B", None)):
        for f in V:
            sc = load(f"scan/{t}_{f}.json")
            o[f"m.{t}.{f}.remove_encode"] = get(sc, "groups", 0, "acc")
            o[f"m.{t}.{f}.baseline"] = get(sc, "baseline", "acc")
            if g is None:
                o[f"m.{t}.{f}.remove_address"] = get(load(f"removal/{t}_{f}.json"), "removed", "acc")
                o[f"m.{t}.{f}.transplant_address"] = final(f"transplant/{t}_{f}_union.json", "G2+G3")
                o[f"m.{t}.{f}.G2_only"] = final(f"transplant/{t}_{f}.json", "G2")
            else:
                o[f"m.{t}.{f}.remove_address"] = get(sc, "groups", g, "acc")
                o[f"m.{t}.{f}.transplant_address"] = final(f"transplant/{t}_{f}.json", f"G{g}")
            tr = load(f"transplant/{t}_{f}.json")
            addr = {f"G{g}"} if g is not None else {"G2", "G3"}
            o[f"m.{t}.{f}.best_other"] = (max(r["ansA"] for r in tr["transplant"] if r["positions"] == "final" and r["name"] not in addr)
                                          if tr else None)


def text_9b(o):
    sc = load("scan/9B_chat8.json")
    o["text.G0.acc"] = get(sc, "groups", 0, "acc")
    base = get(sc, "baseline", "gap")
    o["text.G0.gap_rel"] = sc["groups"][0]["gap"] / base if sc else None
    o["text.G0.att"] = get(sc, "reader_split", "G0", "target")
    o["text.L0.acc"] = get(sc, "layers", 0, "acc")
    o["text.G4.acc"] = get(sc, "groups", 4, "acc")
    o["text.gap.intact"] = base
    o["text.G4.gap"] = get(sc, "groups", 4, "gap")
    o["text.att.intact"] = get(sc, "reader_split", "intact", "target")
    o["text.G4.att"] = get(sc, "reader_split", "G4", "target")
    for f, k in (("list8", "list8"), ("rev8", "rev8"), ("chat8", "chat8")):
        o[f"text.G4_transplant.{k}"] = final(f"transplant/9B_{f}.json", "G4")
        o[f"text.G4_removal.{k}"] = get(load(f"scan/9B_{f}.json"), "groups", 4, "acc")
    tr = load("transplant/9B_chat8.json")
    o["text.G4_dict"] = next((r["ansA"] for r in tr["transplant"] if r["name"] == "G4" and r["positions"] == "dict"), None) if tr else None
    o["text.others_transplant_max"] = (max(r["ansA"] for r in tr["transplant"] if r["positions"] == "final" and r["name"] != "G4") if tr else None)
    sel = {(s["group"], s["q"], s["k"]): s for s in sc["selectivity"]} if sc else {}
    o["text.sel.intact"] = get(sel, ("G4", "intact", "intact"), "selectivity")
    o["text.sel.query_noG4"] = get(sel, ("G4", "G4", "intact"), "selectivity")
    o["text.sel.keys_noG4"] = get(sel, ("G4", "intact", "G4"), "selectivity")
    o["text.top1.query_noG4"] = get(sel, ("G4", "G4", "intact"), "top1")
    o["text.top1.keys_noG4"] = get(sel, ("G4", "intact", "G4"), "top1")
    c = load("construction/9B_chat8.json")
    for k in ("out_final", "state_G4", "state_G3", "state_G0", "input_final"):
        o[f"text.construct.{k}"] = get(c, "rows", k, "ansA")
    for g in (1, 2, 3):
        o[f"text.G{g}.att"] = get(sc, "reader_split", f"G{g}", "target")
    for k in ("up_noG1", "up_noG2", "up_noG3", "up_noG1-3"):
        o[f"text.construct.{k}"] = get(c, "rows", k, "ansA")
    o["text.removal.0.8B_block"] = get(load("scan/0.8B_chat8.json"), "groups", 3, "acc")
    o["text.removal.G1B_block"] = get(load("scan/G1B_chat8.json"), "groups", 2, "acc")


def truncation(o):
    P = load("truncation/9B.json"); K = load("knowledge/9B.json")
    if P is None or K is None:
        return
    TASKS = K["meta"]["tasks"]; CH = float(np.mean([0.25, 0.25, 0.5, 0.5, 0.25, 0.25]))
    key = lambda r: P["meta"]["n_layers"] - 1 if r["cut"] is None else r["cut"]
    ret = np.array([r["dict_cmean"] for r in sorted(P["rows"], key=key)])
    ret = (ret - 0.125) / (ret[-1] - 0.125)
    kn = np.array([np.mean([r["acc"][t] for t in TASKS]) for r in sorted(K["rows"], key=key)])
    kn = (kn - CH) / (kn[-1] - CH)
    o["text.trunc.L18"], o["text.trunc.L19"] = float(max(ret[18], 0)), float(ret[19])
    o["text.trunc.kept24.retrieval"], o["text.trunc.kept24.knowledge"] = float(ret[23]), float(kn[23])


def collect():
    o = {}
    for f in (truncation, text_9b, table1, table2, table3, single_layer, models):
        f(o)
    return o


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def show(o):
    groups = {}
    for k, v in o.items():
        groups.setdefault(k.split(".")[0], []).append((k, v))
    title = dict(text="In-text numbers", t1="Table 1", t2="Table 2", t3="Table 3", sl="Appendix B", m="Appendix C")
    for g, rows in groups.items():
        print(f"\n## {title[g]}")
        for k, v in rows:
            print(f"{k:42s} {fmt(v)}")


def check(o, tol):
    exp = json.load(open(ROOT / "paper_values.json"))
    bad = [(k, v, o.get(k)) for k, v in exp.items()
           if o.get(k) is None or abs(o[k] - v) > (0.02 * abs(v) if abs(v) > 1 else tol)]
    for k, v, got in bad:
        print(f"{k:42s} paper {v:<10g} results {fmt(got)}")
    print(f"{len(exp) - len(bad)}/{len(exp)} values match")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tol", type=float, default=0.006, help="tolerance on rates (values above 1 are checked within 2%%)")
    a = ap.parse_args()
    o = collect()
    if a.check:
        sys.exit(0 if check(o, a.tol) else 1)
    show(o)
