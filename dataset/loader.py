"""Pavia MAT loader for the clean HAD-MSF release."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    directory: str
    filename: str


PAVIA = DatasetSpec("Pavia", "Pavia", "Pavia.mat")


@dataclass
class DatasetBundle:
    name: str
    data: np.ndarray
    gt: np.ndarray
    path: Path
    spec: DatasetSpec

    @property
    def shape(self):
        return self.data.shape

    @property
    def anomaly_count(self):
        return int(np.count_nonzero(self.gt))


def _pick_arrays(mat):
    arrays = {
        key: value for key, value in mat.items()
        if not key.startswith("__") and isinstance(value, np.ndarray)
    }
    cubes = [(key, value) for key, value in arrays.items() if value.ndim == 3]
    maps = [(key, value) for key, value in arrays.items() if value.ndim == 2]
    if not cubes:
        raise ValueError("Pavia.mat must contain one 3-D hyperspectral cube")
    cube_key, cube = max(cubes, key=lambda item: item[1].size)
    for key in ("groundtruth", "ground_truth", "gt", "map", "label", "labels"):
        if key in arrays and arrays[key].ndim == 2:
            return cube, arrays[key], cube_key, key
    matches = [
        (key, value) for key, value in maps
        if value.shape in (cube.shape[:2], cube.shape[1:])
    ]
    if not matches:
        raise ValueError(f"cannot find a 2-D label map for cube {cube.shape}")
    return cube, matches[0][1], cube_key, matches[0][0]


def load_mat(path: Path):
    cube, gt, _, _ = _pick_arrays(sio.loadmat(path))
    if gt.shape == cube.shape[1:]:
        cube = np.transpose(cube, (1, 2, 0))
    if gt.shape != cube.shape[:2]:
        raise ValueError(f"cube shape {cube.shape} and label shape {gt.shape} do not align")
    cube = np.asarray(cube, dtype=np.float32)
    gt = np.asarray(gt > 0, dtype=bool)
    if not np.isfinite(cube).all():
        raise ValueError(f"Pavia data contain NaN or Inf: {path}")
    return cube, gt


def validate_bundle(bundle: DatasetBundle):
    errors = []
    if bundle.data.ndim != 3:
        errors.append(f"cube_ndim={bundle.data.ndim}, expected=3")
    if bundle.gt.ndim != 2:
        errors.append(f"label_ndim={bundle.gt.ndim}, expected=2")
    if bundle.data.shape[:2] != bundle.gt.shape:
        errors.append(f"cube_shape={bundle.data.shape}, label_shape={bundle.gt.shape}")
    if bundle.anomaly_count <= 0:
        errors.append("the label map contains no positive pixels")
    if bundle.anomaly_count >= bundle.gt.size:
        errors.append("the label map contains no background pixels")
    if errors:
        raise ValueError("dataset validation failed: " + "; ".join(errors))


def _find_mat(data_root: Optional[str]) -> Path:
    root = Path(data_root) if data_root else Path(__file__).resolve().parent / "Pavia" / "Pavia.mat"
    if root.is_file():
        if root.suffix.lower() != ".mat":
            raise ValueError(f"data_root must be a .mat file or directory: {root}")
        return root
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    candidates = sorted(root.glob("*.mat"))
    if not candidates:
        candidates = sorted(root.rglob("*.mat"))
    if not candidates:
        raise FileNotFoundError(f"no .mat file found under {root}")
    if len(candidates) > 1:
        names = ", ".join(str(item) for item in candidates)
        raise ValueError("multiple MAT files found; pass the Pavia MAT file: " + names)
    return candidates[0]


def _scene_name(path: Path) -> str:
    return path.stem.replace("_", "-").strip().title() or "Pavia"


def load_dataset(data_root: Optional[str] = None):
    path = _find_mat(data_root)
    data, gt = load_mat(path)
    scene = _scene_name(path)
    if scene.lower() != "pavia":
        raise ValueError(f"This clean release is for Pavia, got {scene}")
    spec = PAVIA
    bundle = DatasetBundle(scene, data, gt, path, spec)
    validate_bundle(bundle)
    return bundle
