"""Backend adapter for the frozen untuned-model decoder.

The API supplies a network dictionary and preprocessing defaults. The evaluated
decoder needs attribute access and its own frozen settings, so adapt both here.
No reference route or label data is accepted or loaded.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from .hmm_decoder import decode_trip as _decode_trip


def decode_trip(candidates, probs, network, gps_xy, times, config):
    """Run the evaluated decoder using the backend's network representation."""
    trip = SimpleNamespace(**network) if isinstance(network, dict) else network
    frozen = json.loads((Path(__file__).parent / 'config.json').read_text(encoding='utf-8'))
    return _decode_trip(candidates, probs, trip, gps_xy, times, frozen)
