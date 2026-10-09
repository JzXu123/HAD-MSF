"""Official-Mamba Pavia runner for HAD-MSF."""
import argparse
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset import load_dataset
from select_bands import SIFDR
from SeT.loss import TotalLoss
from SeT.mask import Mask
from SeT.train import separation_training
from utils import ZScoreNorm


def verify_official_mamba():
    try:
        package = importlib.import_module("mamba_ssm")
        scan = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Official mamba_ssm is required. Install it in a CUDA-enabled "
            "environment with: pip install mamba-ssm causal-conv1d --no-build-isolation"
        ) from exc
    package_path = Path(package.__file__).resolve()
    project_path = Path(__file__).resolve().parent
    if project_path == package_path or project_path in package_path.parents:
        raise RuntimeError(f"mamba_ssm resolves inside the project: {package_path}")
    if scan.selective_scan_fn is not selective_scan_fn:
        raise RuntimeError("mamba_ssm selective_scan_fn is not the official backend")
    if not callable(selective_scan_fn):
        raise RuntimeError("official selective_scan_fn is not callable")
    return package_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested == "auto":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type != "cuda":
        raise RuntimeError("Pavia reproduction requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build")
    return device


def build_parser():
    default_data = Path(__file__).resolve().parent / "dataset" / "Pavia" / "Pavia.mat"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(default_data),
                        help="Pavia MAT file; defaults to the bundled dataset")
    parser.add_argument("--num-bs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lmda", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--vss-depth", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--adawat-kernel-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="results_publishable")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    mamba_path = verify_official_mamba()
    set_seed(args.seed)
    device = choose_device(args.device)
    bundle = load_dataset(args.data_root)
    if bundle.name.lower() != "pavia":
        raise ValueError(f"This clean release is for Pavia, got {bundle.name}")
    if args.num_bs > bundle.data.shape[-1]:
        raise ValueError(f"--num-bs={args.num_bs} exceeds {bundle.data.shape[-1]} bands")

    from model import RAM

    bands = SIFDR(bundle.data, args.num_bs)
    selected = bundle.data[:, :, bands].astype(np.float32)
    selected = ZScoreNorm().fit(selected).transform(selected).astype(np.float32)
    x = torch.from_numpy(selected).to(device)
    model = RAM(
        selected.shape,
        adawat_kernel_size=args.adawat_kernel_size,
        vss_depth=args.vss_depth,
        state_dim=args.state_dim,
    ).to(device)
    criterion = TotalLoss(args.lmda, device=device, eps=args.eps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mask = Mask(selected.shape[:2], device=device)
    output_root = Path(args.output_root) / "Pavia"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(json.dumps({
        "dataset": bundle.name,
        "dataset_file": str(bundle.path),
        "mamba_ssm": str(mamba_path),
        "shape": list(bundle.shape),
        "anomaly_count": bundle.anomaly_count,
        "selected_bands_zero_based": bands.tolist(),
        "band_selection": "SIFDR",
        "num_bs": args.num_bs,
        "epochs": args.epochs,
        "max_iter": args.max_iter,
        "lr": args.lr,
        "lmda": args.lmda,
        "eps": args.eps,
        "vss_depth": args.vss_depth,
        "state_dim": args.state_dim,
        "seed": args.seed,
        "device": str(device),
    }, indent=2), encoding="utf-8")
    _, history, values = separation_training(
        x=x, gt=bundle.gt, model=model, loss=criterion, mask=mask,
        optimizer=optimizer, epochs=args.epochs, max_iter=args.max_iter,
        verbose=not args.quiet, dataset_name=bundle.name, result_root=output_root,
    )
    print(f"mamba_ssm={mamba_path}")
    print(f"saved={output_root / model.name}")
    print(f"best_loss={min(history):.6f}")
    print(f"AUC(Pd,Pf)={values['auc_pd_pf']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
