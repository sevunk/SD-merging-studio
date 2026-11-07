"""Auto-Discovery Registry for Model Architectures in Merging Studio v2.0.

To add a new model architecture in the future:
1. Create `architectures/your_new_model.py` subclassing `ArchitectureSpec`.
2. Add `YourNewModelSpec` to `REGISTERED_ARCHITECTURES` below.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Type
from .base_spec import ArchitectureSpec
from .moss_image import MossImageSpec
from .krea2 import Krea2Spec
from .anima import AnimaSpec
from .minimax_h3 import MinimaxH3Spec

REGISTERED_ARCHITECTURES: List[Type[ArchitectureSpec]] = [
    MossImageSpec,
    Krea2Spec,
    AnimaSpec,
    MinimaxH3Spec,
]

ARCH_REGISTRY: Dict[str, Type[ArchitectureSpec]] = {
    spec.arch_id: spec for spec in REGISTERED_ARCHITECTURES
}


def auto_detect_architecture(all_keys: Sequence[str], filename: str = "") -> Type[ArchitectureSpec]:
    """Automatically detects the model architecture from its state_dict keys or filename."""
    if MossImageSpec.detect(all_keys):
        return MossImageSpec
    if Krea2Spec.detect(all_keys):
        return Krea2Spec
    fname_low = filename.lower()
    if "minimax" in fname_low or "h3" in fname_low:
        return MinimaxH3Spec
    if "anima" in fname_low or AnimaSpec.detect(all_keys):
        return AnimaSpec
    return Krea2Spec
