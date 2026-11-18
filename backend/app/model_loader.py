"""Model and HMM loader for PathSnap API.

Loads the classifier model, feature definitions, and dynamic HMM decoder.
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import lightgbm as lgb

logger = logging.getLogger("pathsnap.api")


@dataclass
class ModelBundle:
    booster: lgb.Booster
    features: List[str]
    config: Dict[str, Any]
    name: str
    version: str


def load_model_bundle(model_dir: Path) -> ModelBundle:
    """Load classifier model, features, and configuration from MODEL_DIR.

    Tolerates CRLF line endings on Windows by reading bytes and normalizing newlines.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir.name}")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found in: {model_dir.name}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    features = cfg.get("features", [])
    if not features:
        raise ValueError("Model config.json contains no 'features' list.")

    model_name = cfg.get("model", "LightGBM")
    model_version = cfg.get("version", "1.0.0")
    if model_version == "1.0.0" and "seeds" in cfg:
        model_version = f"seed-{cfg['seeds'][0]}"

    # Locate the model file (lightgbm_model.txt or any .txt/.model in dir)
    model_txt = model_dir / "lightgbm_model.txt"
    if not model_txt.exists():
        txt_files = list(model_dir.glob("*.txt"))
        if txt_files:
            model_txt = txt_files[0]
        else:
            raise FileNotFoundError("No lightgbm_model.txt found in model directory.")

    # Read binary and convert CRLF to LF to prevent LightGBM tree_sizes byte offset bug on Windows
    with open(model_txt, "rb") as f:
        raw_bytes = f.read().replace(b"\r\n", b"\n")

    model_str = raw_bytes.decode("utf-8")
    booster = lgb.Booster(model_str=model_str)

    logger.info(f"Loaded classifier model: {model_name} ({model_version}), {len(features)} features, {booster.num_trees()} trees")
    return ModelBundle(
        booster=booster,
        features=features,
        config=cfg,
        name=model_name,
        version=model_version,
    )


def load_hmm_decoder(module_spec: Optional[str]) -> Optional[Callable]:
    """Dynamically load the HMM decoder function from HMM_MODULE.

    Expected contract:
        decode_trip(candidates, probs, network, gps_xy, times, config)
        -> {
            "arc_per_point": list[int] or np.ndarray,
            "predicted_route": list[int] or np.ndarray,
            "diagnostics": {
                "fallbacks_unreachable": int,
                "fallbacks_gap": int,
                ...
            }
        }
    with NO reference route argument.
    """
    if not module_spec or not module_spec.strip():
        logger.info("HMM_MODULE is blank: HMM decoding disabled (classifier-only mode).")
        return None

    try:
        spec = module_spec.strip()
        if ":" not in spec:
            raise ValueError(f"Invalid HMM_MODULE specification: '{spec}'. Must be 'module.path:function_name'.")
        mod_name, fn_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            raise TypeError(f"Target '{fn_name}' in '{mod_name}' is not callable.")
        logger.info(f"Loaded HMM decoder: {spec}")
        return fn
    except Exception as e:
        logger.warning(f"Could not load HMM decoder from '{module_spec}': {e}. Falling back to classifier-only mode.")
        return None
