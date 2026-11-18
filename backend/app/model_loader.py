"""Model and HMM loader for PathSnap API.

Loads the classifier model, feature definitions, and dynamic HMM decoder.
"""
from __future__ import annotations

import importlib
import hashlib
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import lightgbm as lgb

logger = logging.getLogger("pathsnap.api")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FINAL_HMM_MODULE = "trained_models.hmm.api_adapter:decode_trip"


@dataclass
class ModelBundle:
    booster: lgb.Booster
    features: List[str]
    config: Dict[str, Any]
    name: str
    version: str
    model_hash: str


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
    if "variant" in cfg and "seed" in cfg:
        model_version = f"{cfg['variant']}-seed-{cfg['seed']}"
    elif model_version == "1.0.0" and "seeds" in cfg:
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
        file_bytes = f.read()
    model_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    raw_bytes = file_bytes.replace(b"\r\n", b"\n")

    model_str = raw_bytes.decode("utf-8")
    booster = lgb.Booster(model_str=model_str)

    logger.info(f"Loaded classifier model: {model_name} ({model_version}), {len(features)} features, {booster.num_trees()} trees")
    return ModelBundle(
        booster=booster,
        features=features,
        config=cfg,
        name=model_name,
        version=model_version,
        model_hash=model_hash,
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
        # Never let a test stub or another implementation be presented as the
        # evaluated final HMM. Classifier-only mode remains available explicitly.
        if spec != FINAL_HMM_MODULE:
            raise ValueError("HMM_MODULE must name the final decoder from COMPARISON_REPORT.md")
        if ":" not in spec:
            raise ValueError(f"Invalid HMM_MODULE specification: '{spec}'. Must be 'module.path:function_name'.")
        mod_name, fn_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            raise TypeError(f"Target '{fn_name}' in '{mod_name}' is not callable.")
        if list(inspect.signature(fn).parameters) != ["candidates", "probs", "network", "gps_xy", "times", "config"]:
            raise ValueError("HMM decoder has an incompatible signature")
        logger.info(f"Loaded HMM decoder: {spec}")
        return fn
    except Exception as e:
        raise RuntimeError(f"Configured HMM decoder could not be loaded: {e}") from e


def load_hmm_config(path: Optional[Path]) -> tuple[dict, str]:
    """Validate the exact frozen config used by the immutable API adapter."""
    if path is None or not path.is_file():
        raise ValueError("HMM_CONFIG must point to an existing decoder config.json")
    expected = REPO_ROOT / "trained_models" / "hmm" / "config.json"
    if path.resolve() != expected.resolve():
        raise ValueError("HMM_CONFIG must point to trained_models/hmm/config.json for the final adapter")
    raw = path.read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HMM_CONFIG is not valid JSON") from exc
    required = {"alpha", "eps", "beta", "u_turn_penalty", "gap_beta_factor", "gap_seconds", "v_max_mps", "min_cutoff_m", "hmm_top_k", "backward_tolerance_m", "unreachable_logp"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError("HMM_CONFIG is missing required decoder settings")
    return config, hashlib.sha256(raw).hexdigest()[:12]
