"""Figures of the paper, from results/ (no GPU needed): vector PDF in figures/, TeX Gyre Termes text.

    python make_figures.py [fig2 fig3 fig4 scaling_readers]

Fig. 2  truncation curve                   results/truncation/9B.json, results/knowledge/9B.json
Fig. 3  removal and transplant per block   results/scan/9B_chat8.json, results/transplant/9B_chat8.json
Fig. 4  (a) reader selectivity             results/scan/9B_{chat8,list8,rev8}.json
        (b, c) what carries the address    results/construction/9B_{chat8,list8,rev8}.json
Appendix D  readers guided by each block   results/transplant/{27B,35B}_chat8_union.json
"""
import json, sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
for f in glob.glob(str(ROOT / "assets" / "fonts" / "*.otf")):
    fm.fontManager.addfont(f)

INK, MUTED, GRID = "#222222", "#666666", "#e4e4e4"
BLUE, ORANGE, GREEN = "#2a77d8", "#e8662f", "#1aa477"
SHADE = "#efeeea"
plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": "#444444", "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "hatch.linewidth": 0.6,
})
TEXTW = 6.5


def load(p):
    return json.load(open(ROOT / "results" / p))


def clean(ax, grid_y=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_y:
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    print("wrote", OUT / f"{name}.pdf")


# ===================================================================== figure 2
KN_TASKS = ["arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa", "hellaswag"]
KN_CHANCE = [0.25, 0.25, 0.5, 0.5, 0.25, 0.25]


def trunc_curves(tag):
    """Normalized truncation curves.
    Retrieval: choice among the dictionary's 8 values, mean over chat8/chat16/list8/rev8/long512
    (dict_cmean), chance 1/8. Knowledge: mean accuracy of the six lm-eval tasks, normalized with the
    mean chance. Normalized = (x - chance) / (x_full - chance), below chance shown as 0."""
    P = load(f"truncation/{tag}.json")
    K = load(f"knowledge/{tag}.json")
    n = P["meta"]["n_layers"]
    key = lambda r: n - 1 if r["cut"] is None else r["cut"]
    P = sorted(P["rows"], key=key); K = sorted(K["rows"], key=key)
    ret = np.array([r["dict_cmean"] for r in P])
    ret = np.clip((ret - 0.125) / (ret[-1] - 0.125), 0, None)
    kn = np.array([np.mean([r["acc"][t] for t in KN_TASKS]) for r in K])
    ch = float(np.mean(KN_CHANCE))
    kn = np.clip((kn - ch) / (kn[-1] - ch), 0, None)
    return np.arange(n), ret, kn


def trunc_figure(tag, reader, address, out, annot_xy=None):
    L, ret, kn = trunc_curves(tag)
    n = len(L)
    fig = plt.figure(figsize=(TEXTW, 2.2))
    ax = fig.add_axes([0.075, 0.33, 0.915, 0.65])
    st = fig.add_axes([0.075, 0.235, 0.915, 0.07])
    HL = "#dce9f9"                                    # light blue highlight for the reader layer
    ax.axvspan(reader - 0.45, reader + 0.45, color=HL, zorder=0, lw=0)
    ax.plot(L, ret, "-o", color=BLUE, lw=1.4, ms=3.2, mec="white", mew=0.5, zorder=3,
            label="Retrieval (dictionary lookup, choice among 8 values)")
    ax.plot(L, kn, "-o", color=ORANGE, lw=1.4, ms=3.2, mec="white", mew=0.5, zorder=3,
            label="Knowledge (mean of six benchmarks)")
    tx, ty = annot_xy or (reader - 3.8, 0.60)
    ax.annotate(f"Retrieval appears at L{reader}", xy=(reader, ret[reader]), xytext=(tx, ty), fontsize=8,
                color=INK, ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.7, shrinkA=1, shrinkB=3, mutation_scale=7))
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(range(n)); ax.set_xticklabels([]); ax.tick_params(axis="x", length=2.2, width=0.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Normalized accuracy")
    clean(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
    # layer strip
    st.set_xlim(-0.6, n - 0.4); st.set_ylim(0, 1); st.axis("off")
    for i in range(n):
        sm = (i + 1) % 4 == 0
        st.add_patch(Rectangle((i - 0.45, 0), 0.9, 1, fc="#4a4a4a" if sm else "#dcdbd6", lw=0))
        if i == reader:
            st.add_patch(Rectangle((i - 0.55, -0.15), 1.1, 1.3, fc="none", ec="#6ea4e8", lw=1.1, zorder=3, clip_on=False))
        st.text(i, -0.35, str(i), ha="center", va="top", fontsize=6.8, color=MUTED)
    for lo, hi, lab in [(0, 2, "Encode"), (address[0], address[1], "Address"), (reader, reader, "Reader")]:
        y = -1.55
        st.plot([lo - 0.4, lo - 0.4, hi + 0.4, hi + 0.4], [y + 0.25, y, y, y + 0.25], color="#555555", lw=0.6,
                clip_on=False)
        if lab == "Reader":
            st.text(lo - 0.4, y - 0.2, f"Reader L{reader}", ha="left", va="top", fontsize=8, color=INK)
        else:
            st.text((lo + hi) / 2 - (0.3 if lab == "Address" else 0), y - 0.2, lab, ha="center", va="top",
                    fontsize=8, color=INK)
    st.text((n - 1) / 2, -2.85, "Last layer kept ($L$)", ha="center", va="top", fontsize=8.5, color=INK)
    for k, (fc, lab) in enumerate([("#4a4a4a", "softmax"), ("#dcdbd6", "linear")]):
        x0 = max(n - 7.4, reader + 3.7) + k * 3.6 * n / 32
        st.add_patch(Rectangle((x0, -1.75), 0.75 * n / 32, 0.6, fc=fc, lw=0, clip_on=False))
        st.text(x0 + 1.0 * n / 32, -1.45, lab, va="center", fontsize=7.5, color=MUTED)
    save(fig, out)


def fig2():
    trunc_figure("9B", reader=19, address=(16, 18), out="prune_curve_9B", annot_xy=(15.2, 0.60))


# ===================================================================== figure 3
def fig3():
    base = load("scan/9B_chat8.json")
    g0 = base["baseline"]["gap"]
    acc = [g["acc"] for g in base["groups"]]
    gap = [g["gap"] / g0 for g in base["groups"]]
    rs = base["reader_split"]
    att = [rs[f"G{i}"]["target"] for i in range(8)]
    tr = load("transplant/9B_chat8.json")["transplant"]
    fin = {r["group"]: r for r in tr if r["positions"] == "final"}
    ansA = [fin[i]["ansA"] for i in range(8)]
    attA = [fin[i]["attA"] for i in range(8)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 1.6), gridspec_kw=dict(wspace=0.10))
    x = np.arange(8); w = 0.27
    for ax in (a1, a2):
        ax.axvspan(-0.5, 0.5, color=SHADE, lw=0, zorder=0)
        ax.axvspan(3.5, 4.5, color=SHADE, lw=0, zorder=0)
        ax.axvline(4.5, color="#777777", lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
        ax.text(0, 1.125, "Encode", ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.text(4.42, 1.125, "Address", ha="right", va="center", fontsize=7.5, fontweight="bold")
        ax.text(4.58, 1.125, "reader L19", ha="left", va="center", fontsize=7, color=MUTED)
        ax.set_xticks(x); ax.set_xticklabels([f"$G_{i}$" for i in range(8)])
        ax.set_ylim(0, 1.2); ax.set_yticks([0, 0.5, 1.0]); ax.set_xlim(-0.6, 7.6)
        ax.tick_params(axis="x", length=2.5)
        clean(ax)
    a1.bar(x - w, acc, w, color=BLUE, label="accuracy", zorder=2)
    a1.bar(x, gap, w, color=ORANGE, label="logit gap (relative)", zorder=2)
    a1.bar(x + w, att, w, color=GREEN, label="reader attention", zorder=2)
    a2.bar(x - w / 2, ansA, w, color=BLUE, label="answers the donor's question", zorder=2)
    a2.bar(x + w / 2, attA, w, color=GREEN, label="reader attention on donor's entry", zorder=2)
    a2.set_yticklabels([])
    a1.set_title("(a) Removal", loc="left", pad=13)
    a2.set_title("(b) Transplant at the final position", loc="left", pad=13)
    for ax in (a1, a2):
        ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 0.985), ncol=3, frameon=False, handlelength=0.9,
                  handletextpad=0.35, columnspacing=0.9, fontsize=7, borderaxespad=0)
    save(fig, "circuit_9B")


# ===================================================================== figure 4
V = ["chat8", "list8", "rev8"]
VLAB = {"chat8": "chat8", "list8": "list8", "rev8": "rev8"}


def bar_style(i, color):
    """chat8: solid; list8: light; rev8: light with hatching (the 'shadow')."""
    if i == 0:
        return dict(facecolor=color, edgecolor="none")
    if i == 1:
        return dict(facecolor=color, alpha=0.45, edgecolor="none")
    return dict(facecolor="white", edgecolor=color, hatch="//////", lw=0.6)


def sel_cells(path, g_addr, g_up):
    s = {(x["group"], x["q"], x["k"]): x for x in load(path)["selectivity"]}
    c = [s[(g_addr, "intact", "intact")], s[(g_addr, "intact", g_addr)], s[(g_addr, g_addr, "intact")],
         s[(g_up, g_up, "intact")]]
    return [x["selectivity"] for x in c], [x["top1"] for x in c]


def fig4():
    sel = {v: sel_cells(f"scan/9B_{v}.json", "G4", "G3") for v in V}
    bc = {v: load(f"construction/9B_{v}.json")["rows"] for v in V}

    fig = plt.figure(figsize=(TEXTW, 1.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.95], wspace=0.62)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    w = 0.26

    # (a) selectivity
    cats = ["intact", "keys\nw/o $G_4$", "query\nw/o $G_4$", "query\nw/o $G_3$"]
    x = np.arange(4)
    for i, v in enumerate(V):
        s, t1 = sel[v]
        bars = a.bar(x + (i - 1) * w, s, w, zorder=2, label=VLAB[v], **bar_style(i, BLUE))
        for xi, (si, ti) in enumerate(zip(s, t1)):
            a.text(x[xi] + (i - 1) * w, si + 0.12, f"{ti:.2f}", ha="center", va="bottom", fontsize=6,
                   color=MUTED, rotation=90)
    a.set_xticks(x); a.set_xticklabels(cats, fontsize=7.5); a.tick_params(axis="x", length=2.5)
    a.set_ylabel("Reader selectivity"); a.set_ylim(0, 8.0); a.set_yticks([0, 3, 6])
    a.set_title("(a) Query or keys?", loc="left")
    clean(a)

    # (b) what carries the address (horizontal)
    rows = [("out_final", "output (final)", BLUE), ("state_G4", "state of $G_4$", GREEN),
            ("state_G3", "state of $G_3$", GREEN), ("state_G0", "state of $G_0$", GREEN),
            ("input_final", "input (final)", BLUE)]
    y = np.arange(len(rows))[::-1]
    for i, v in enumerate(V):
        for yi, (k, lab, col) in zip(y, rows):
            val = bc[v][k]["ansA"]
            b.barh(yi - (i - 1) * w, val, w, zorder=2, **bar_style(i, col))
    for yi, (k, lab, col) in zip(y, rows):
        if max(bc[v][k]["ansA"] for v in V) < 0.02:
            b.text(0.02, yi, "0.00", va="center", fontsize=6.5, color=MUTED)
    b.set_yticks(y); b.set_yticklabels([r[1] for r in rows], fontsize=7.5); b.tick_params(axis="y", length=2.5)
    b.set_xlim(0, 1.05); b.set_xticks([0, 0.5, 1.0]); b.set_xlabel("Redirection rate")
    b.set_title("(b) What carries the address?", loc="left")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); b.set_axisbelow(True)

    # (c) Address computed without upstream blocks
    keys = [("up_noG1", "w/o\n$G_1$"), ("up_noG2", "w/o\n$G_2$"), ("up_noG3", "w/o\n$G_3$"),
            ("up_noG1-3", "w/o\n$G_1$–$G_3$")]
    x = np.arange(4)
    for i, v in enumerate(V):
        vals = [bc[v][k]["ansA"] for k, _ in keys]
        c.bar(x + (i - 1) * w, vals, w, zorder=2, **bar_style(i, BLUE))
    if max(bc[v]["up_noG1-3"]["ansA"] for v in V) < 0.02:
        c.text(3, 0.03, "0.00", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    c.set_xticks(x); c.set_xticklabels([k[1] for k in keys], fontsize=7.5); c.tick_params(axis="x", length=2.5)
    c.set_ylim(0, 1.05); c.set_yticks([0, 0.5, 1.0]); c.set_ylabel("Redirection rate")
    c.set_title("(c) Address without upstream", loc="left")
    clean(c)

    # shared format legend
    from matplotlib.patches import Patch
    hs = [Patch(label=VLAB[v], **bar_style(i, "#555555")) for i, v in enumerate(V)]
    b.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 0.02), ncol=1, frameon=False,
             handlelength=1.6, labelspacing=0.3, fontsize=7.5, borderaxespad=0.1)
    save(fig, "construction_9B")


# ===================================================================== appendix figure
R27 = ["39,15", "43,22", "51,20", "55,22", "59,3"]
ROWS27 = ["G8", "G9", "G10", "G11", "G12", "G13", "G14", "G8+G11+G12", "G8+G10+G11+G12", "G8+G9+G10+G11+G12"]
LAB27 = ["$G_8$", "$G_9$", "$G_{10}$", "$G_{11}$", "$G_{12}$", "$G_{13}$", "$G_{14}$", "$G_8{+}G_{11}{+}G_{12}$",
         "$+\\,G_{10}$", "$G_8$–$G_{12}$"]
R35 = ["19,4", "27,6", "31,4", "35,13"]
ROWS35 = ["G4", "G5", "G6", "G7", "G8", "G6+G7", "G6+G7+G8", "G4+G5+G6+G7+G8"]
LAB35 = ["$G_4$", "$G_5$", "$G_6$", "$G_7$", "$G_8$", "$G_6{+}G_7$", "$G_6$–$G_8$", "$G_4$–$G_8$"]


def _matrix(path, readers, rows):
    """Attention of each tracked reader on the donor's entry, per transplanted block set."""
    d = load(path)
    main = ",".join(map(str, d["meta"]["reader"]))
    got = {r["name"]: r for r in d["transplant"] if r["positions"] == "final"}
    key = lambda rd: "attA" if rd == main else "L{}H{}_A".format(*rd.split(","))
    return np.array([[got[nm][key(rd)] for rd in readers] for nm in rows])


def _heat(ax, M, rows, cols, title, n_single):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.6, color="white" if M[i, j] > 0.5 else INK)
    ax.axhline(n_single - 0.5, color=INK, lw=0.6)
    ax.set_title(title, loc="left", fontsize=9)
    return im


def scaling_readers():
    M27 = _matrix("transplant/27B_chat8_union.json", R27, ROWS27)
    M35 = _matrix("transplant/35B_chat8_union.json", R35, ROWS35)
    fig = plt.figure(figsize=(TEXTW * 0.78, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.2], wspace=0.55, left=0.17, right=0.97, top=0.9, bottom=0.1)
    a2 = fig.add_subplot(gs[0, 0]); a3 = fig.add_subplot(gs[0, 1])
    _heat(a2, M27, LAB27, [f"L{r.split(',')[0]}" for r in R27], "(a) Qwen3.5-27B", 7)
    im = _heat(a3, M35, LAB35, [f"L{r.split(',')[0]}" for r in R35], "(b) Qwen3.5-35B-A3B", 5)
    cb = fig.colorbar(im, ax=[a2, a3], fraction=0.04, pad=0.04, aspect=30)
    cb.ax.tick_params(labelsize=7, length=0); cb.outline.set_visible(False)
    cb.set_label("Reader attention on the donor's entry", fontsize=7.5)
    save(fig, "scaling_readers")


FIGS = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "scaling_readers": scaling_readers}
if __name__ == "__main__":
    for w in sys.argv[1:] or FIGS:
        FIGS[w]()
