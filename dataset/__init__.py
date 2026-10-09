"""Pavia dataset API."""
from .loader import PAVIA, DatasetBundle, DatasetSpec, load_dataset, load_mat, validate_bundle

__all__ = [
    "PAVIA",
    "DatasetBundle",
    "DatasetSpec",
    "load_dataset",
    "load_mat",
    "validate_bundle",
]
