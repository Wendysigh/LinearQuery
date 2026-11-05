"""Run the experiments behind the paper's figures and tables on a pool of GPUs.

    python reproduce.py --list
    python reproduce.py fig3 table3 --gpus 0,1,2,3
    python reproduce.py all --gpus 0,1,2,3,4,5,6,7

Finished jobs (the output file exists) are skipped, so a run can be resumed. Logs go to logs/.
"""
import argparse, os, queue, subprocess, sys, threading, time
from linquery.config import ROOT, FORMATS, MODELS

SMALL = ("0.8B", "9B", "G1B", "G7B")
UNION27 = "8;9;10;11;12;13;14;8,11,12;8,10,11,12;8-12"
UNION35 = "4;5;6;7;8;6,7;6-8;4-8"
LAYERS = {"0.8B": "12,13,14", "9B": "16,17,18", "G1B": "16,17,18,19,20,21,22,23,24"}
SHARDS = 4


def job(out, module, *args):
    return dict(out=out, cmd=[sys.executable, "-m", f"experiments.{module}", *args])


def m(t, f):
    return ["--model", t, "--format", f]


def scan_9b():
    qk = {"chat8": "0,1,2,3,4,5,6,7", "list8": "3,4", "rev8": "3,4"}
    return [job(f"scan/9B_{f}.json", "scan", *m("9B", f), "--qk_groups", qk[f]) for f in FORMATS]


def scan_others():
    return [job(f"scan/{t}_{f}.json", "scan", *m(t, f)) for t in ("0.8B", "G1B", "G7B") for f in FORMATS]


def singles(tags):
    out = []
    for t in tags:
        for f in FORMATS:
            pos = ["--positions", "final,dict"] if (t, f) == ("9B", "chat8") else []
            out.append(job(f"transplant/{t}_{f}.json", "transplant", *m(t, f), *pos))
    return out


def unions():
    out = [job(f"transplant/G7B_{f}_union.json", "transplant", *m("G7B", f), "--union", "2,3", "--name", "union") for f in FORMATS]
    for t, u in (("27B", UNION27), ("35B", UNION35)):
        out += [job(f"transplant/{t}_{f}_union.json", "transplant", *m(t, f), "--union", u, "--name", "union") for f in FORMATS]
    return out


def removal(tags):
    return [job(f"removal/{t}_{f}.json", "remove_blocks", *m(t, f)) for t in tags for f in FORMATS]


def layers():
    return [job(f"transplant/{t}_chat8_layers.json", "transplant", *m(t, "chat8"), "--layers", v, "--name", "layers")
            for t, v in LAYERS.items()]


def fig2():
    out = [job(f"truncation/9B_shard{k}of{SHARDS}.json", "truncation", "--model", "9B", "--shard", f"{k}/{SHARDS}") for k in range(1, SHARDS + 1)]
    out += [job(f"knowledge/9B_shard{k}of{SHARDS}.json", "knowledge", "--model", "9B", "--shard", f"{k}/{SHARDS}") for k in range(1, SHARDS + 1)]
    return out


def table1():
    return [job("encode/swap_9B.json", "encode_swap", "--model", "9B"),
            job("encode/replace_9B.json", "encode_replace", "--model", "9B")]


def fig4():
    return scan_9b() + [job(f"construction/9B_{f}.json", "construction", "--format", f) for f in FORMATS]


def table2():
    out = []
    for d, r in (("0.8B", "G1B"), ("G1B", "0.8B")):
        out += [job(f"transfer/output_{d}_to_{r}_{f}.json", "transfer_output", "--donor", d, "--recv", r, "--format", f) for f in FORMATS]
    O = "transfer_output"
    out += [job("transfer/output_9B_to_G7B_chat8_G2G3.json", O, "--donor", "9B", "--recv", "G7B", "--recv_query", "16-24,26-34", "--name", "G2G3"),
            job("transfer/output_9B_to_G7B_chat8_G2.json", O, "--donor", "9B", "--recv", "G7B", "--name", "G2"),
            job("transfer/output_G7B_to_9B_chat8_G2G3.json", O, "--donor", "G7B", "--recv", "9B", "--combine_src", "2+3", "--name", "G2G3")]
    B = "transfer_block"
    out += [job("transfer/block_0.8B_into_G1B_chat8.json", B, "--donor", "0.8B", "--recv", "G1B", "--slots", "16-24"),
            job("transfer/block_G1B_into_0.8B_chat8.json", B, "--donor", "G1B", "--recv", "0.8B", "--slots", "12-14"),
            job("transfer/block_9B_into_G7B_chat8.json", B, "--donor", "9B", "--recv", "G7B", "--slots", "16-24,26-34"),
            job("transfer/block_G7B_into_9B_chat8.json", B, "--donor", "G7B", "--recv", "9B", "--slots", "16-18", "--cut", "19",
                "--donor_blocks", "2,3", "--donor_custom", "seq:16-24+26-34")]
    return out


ITEMS = {
    "fig2": fig2,
    "fig3": lambda: scan_9b() + singles(["9B"]),
    "table1": table1,
    "fig4": fig4,
    "table2": table2,
    "table3": lambda: removal(MODELS) + singles(["0.8B", "2B", "4B", "9B"]) + unions(),
    "single_layer": lambda: scan_9b() + scan_others() + singles(SMALL) + layers(),
    "models": lambda: scan_9b() + scan_others() + singles(SMALL) + removal(["G7B"]) + unions(),
    "scaling_readers": unions,
}
MERGE = [("truncation/9B.json", "truncation"), ("knowledge/9B.json", "knowledge")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="*", help=f"all, or any of: {', '.join(ITEMS)}")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--list", action="store_true", help="print the commands and exit")
    a = ap.parse_args()
    names = list(ITEMS) if "all" in a.items or not a.items else a.items
    jobs = {}
    for n in names:
        for j in ITEMS[n]():
            jobs[j["out"]] = j
    merged = {"truncation", "knowledge"}                # shards are deleted once merged
    done = lambda j: (ROOT / "results" / j["out"]).exists() or (
        j["out"].split("/")[0] in merged and (ROOT / "results" / (j["out"].split("_shard")[0] + ".json")).exists())
    todo = [j for j in jobs.values() if not done(j)]
    if a.list:
        for j in jobs.values():
            state = "todo" if j in todo else "done"
            print(f"[{state}] python -m {' '.join(j['cmd'][2:])}")
        return
    os.makedirs(ROOT / "logs", exist_ok=True)
    q = queue.Queue()
    # large models first, so a full card pool is not blocked by them at the end
    size = lambda j: -max(float(t[:-1]) for t in j["out"].replace("_", "/").split("/") + ["0B"]
                          if t[:-1].replace(".", "").isdigit() and t.endswith("B"))
    for j in sorted(todo, key=size):
        q.put(j)
    failed = []

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            log = ROOT / "logs" / (j["out"].replace("/", "_").replace(".json", ".log"))
            t0 = time.time()
            print(f"{time.strftime('%H:%M')} gpu {gpu} start {j['out']}", flush=True)
            with open(log, "w") as f:
                rc = subprocess.call(j["cmd"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
            print(f"{time.strftime('%H:%M')} gpu {gpu} {'done' if rc == 0 else 'FAILED'} {j['out']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if rc:
                failed.append(j["out"])

    ts = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if "fig2" in names and not failed:
        for out, mod in MERGE:
            if not (ROOT / "results" / out).exists():
                subprocess.check_call([sys.executable, "-m", f"experiments.{mod}", "--model", "9B", "--merge"], cwd=ROOT)
            for f in (ROOT / "results" / mod).glob("9B_shard*.json"):
                f.unlink()
    print(f"{len(todo) - len(failed)} jobs finished, {len(failed)} failed" + (f": {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
