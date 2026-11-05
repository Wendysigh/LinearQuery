# Reproducing the paper

Interventions are forward hooks on the token-mixer outputs of frozen public checkpoints; no weights are
modified. All models run in bf16 with eager attention. Install the dependencies as in the
[README](README.md#getting-started).

## Models

Checkpoints download from Hugging Face on first use. To use local copies, place or symlink them as
`models/<name>`, for example `models/Qwen3.5-9B`.

| Tag | Paper name | Hugging Face repository |
|---|---|---|
| `0.8B` | Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` |
| `2B` | Qwen3.5-2B | `Qwen/Qwen3.5-2B` |
| `4B` | Qwen3.5-4B | `Qwen/Qwen3.5-4B` |
| `9B` | Qwen3.5-9B | `Qwen/Qwen3.5-9B` |
| `27B` | Qwen3.5-27B | `Qwen/Qwen3.5-27B` |
| `35B` | Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B` |
| `G1B` | Granite4.0-1B | `ibm-granite/granite-4.0-h-1b` |
| `G7B` | Granite4.0-7B | `ibm-granite/granite-4.0-h-tiny` |

## From the saved results (no GPU)

`results/` holds the output of every experiment the paper cites.

```bash
python make_figures.py          # figures/*.pdf
python make_tables.py           # every table and in-text number
python make_tables.py --check   # compare with the values printed in the paper
```

## Rerunning the experiments

```bash
python reproduce.py --list                        # every command, and whether its output exists
python reproduce.py table3 --gpus 0,1,2,3         # one paper item
python reproduce.py all --gpus 0,1,2,3,4,5,6,7    # everything
```

Jobs whose output file exists are skipped, so delete a file in `results/` to rerun it. Logs go to
`logs/`. The full set is 92 jobs and takes about 35 minutes on 8 GPUs. A card runs one job at a time, and
the largest model needs about 70 GB of GPU memory.

## Paper item to experiment

| Paper item | `reproduce.py` | Experiments | Results |
|---|---|---|---|
| Fig. 2, truncation | `fig2` | `truncation`, `knowledge` | `truncation/`, `knowledge/` |
| Fig. 3, removal and transplant | `fig3` | `scan`, `transplant` | `scan/9B_chat8.json`, `transplant/9B_chat8.json` |
| Table 1, Encode | `table1` | `encode_swap`, `encode_replace` | `encode/` |
| Fig. 4, Address construction | `fig4` | `scan`, `construction` | `scan/9B_*.json`, `construction/` |
| Table 2, cross-architecture transfer | `table2` | `transfer_output`, `transfer_block` | `transfer/` |
| Table 3, scaling | `table3` | `remove_blocks`, `transplant` | `removal/`, `transplant/` |
| Appendix B, single layers | `single_layer` | `scan`, `transplant --layers` | `scan/`, `transplant/*_layers.json` |
| Appendix C, models and formats | `models` | `scan`, `remove_blocks`, `transplant` | `scan/`, `removal/`, `transplant/` |
| Appendix D, readers in larger models | `scaling_readers` | `transplant --union` | `transplant/{27B,35B}_chat8_union.json` |

Each job can also be run on its own, for example:

```bash
python -m experiments.transplant --model 9B --format chat8
```

## Task

Each prompt holds a dictionary of eight key-value pairs and a question about one key. The formats are
`chat8`, `list8` and `rev8` (see `linquery/prompts.py`). Each format has 100 prompt pairs:

- Accuracy counts both the clean and the corrupted prompt of each pair.
- Transplant uses donor and host prompts that share a dictionary and differ only in the queried key.
- The retrieval curve in Fig. 2 also includes `chat16` and `long512`, and scores the choice among the
  dictionary's values.
