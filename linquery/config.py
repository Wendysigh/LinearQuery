"""Model registry (configs/models.yaml) and repository paths."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))
FORMATS = ("chat8", "list8", "rev8")


def model_path(tag):
    """models/<name> when it exists locally, otherwise the Hugging Face repository id."""
    repo = MODELS[tag]["id"]
    local = ROOT / "models" / repo.split("/")[-1]
    return str(local) if local.exists() else repo


def reader(tag, fmt):
    return tuple(MODELS[tag]["readers"][fmt])


def group_size(tag):
    return MODELS[tag]["group_size"]


def address_layers(tag):
    """The Address layers of the first Address block: '16-24' -> [16, ..., 24]."""
    lo, hi = map(int, str(MODELS[tag]["address"]).split("-"))
    return list(range(lo, hi + 1))


def out_path(*parts):
    p = RESULTS.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
