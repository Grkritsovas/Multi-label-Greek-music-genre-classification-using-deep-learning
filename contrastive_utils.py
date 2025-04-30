# -------------  contrastive_utils.py  -----------------
import json
import os


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_global_stats(segment_length: int,
                      json_dir: str = "json/global_norm_stats") -> dict:

    fname = f"global_norm_stats_{segment_length}s_contrastive.json"
    data  = _load_json(os.path.join(json_dir, fname))
    return {"mean": data["mean"], "std": data["std"]}


def load_time_axis_stats(segment_length: int,
                         json_dir: str = "json/average_time_axis") -> dict:

    fname = f"average_time_axis_{segment_length}s.json"
    return _load_json(os.path.join(json_dir, fname))
