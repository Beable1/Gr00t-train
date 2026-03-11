#!/usr/bin/env python3
"""
Convenience launcher for fine-tuning NVIDIA Isaac GR00T (via LeRobot) on
your R1 dataset `Beable/dexterious_ee`.

This script wraps the `lerobot-train` CLI with safe defaults so you can run:

    python train_groot_beable.py --output_dir ./outputs/groot_beable

on a cloud GPU machine.

Key choices to avoid previous issues:
- Uses policy.type=groot (GR00T head) instead of plain diffusion.
- Disables tuning of the heavy diffusion model (tune_diffusion_model=false)
  so only the action head is adapted to your robot.
- Disables image cropping (policy.crop_shape=null) so the model always
  sees the full 360x640 frame (no tiny 84x84 crops).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import List


def build_single_gpu_cmd(args: argparse.Namespace) -> List[str]:
    """Build a single‑GPU lerobot-train command."""
    cmd: list[str] = [
        "lerobot-train",
        f"--output_dir={args.output_dir}",
        "--save_checkpoint=true",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        "--eval_freq=0",
        "--save_freq=20000",
        "--log_freq=200",
        # GR00T policy
        "--policy.type=groot",
        "--policy.push_to_hub=false",
        "--policy.tune_diffusion_model=false",
        # Your dataset on Hugging Face
        f"--dataset.repo_id={args.dataset_repo_id}",
        # Make sure images use ImageNet stats and torchcodec backend
        "--dataset.use_imagenet_stats=true",
        "--dataset.video_backend=torchcodec",
        # IMPORTANT: disable cropping – use full 360x640 image
        "--policy.crop_shape=null",
        # Optional but useful metadata
        f"--job_name={args.job_name}",
    ]
    return cmd


def build_multi_gpu_cmd(args: argparse.Namespace) -> List[str]:
    """Wrap the single‑GPU command with accelerate for multi‑GPU training."""
    base_cmd = build_single_gpu_cmd(args)
    # Replace the initial "lerobot-train" with the path from which(lerobot-train)
    if base_cmd[0] != "lerobot-train":
        raise RuntimeError("Unexpected base command layout.")

    lerobot_train_path = os.environ.get("LEROBOT_TRAIN_BIN", "lerobot-train")
    inner_cmd = [lerobot_train_path] + base_cmd[1:]

    cmd: list[str] = [
        "accelerate",
        "launch",
        "--multi_gpu",
        f"--num_processes={args.num_gpus}",
        lerobot_train_path,
    ] + inner_cmd[1:]
    return cmd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine‑tune NVIDIA Isaac GR00T (via LeRobot) on Beable/dexterious_ee."
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="Beable/dexterious_ee",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/groot_beable",
        help="Where to store training outputs and checkpoints.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Global batch size.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs. Use >1 together with accelerate for multi‑GPU.",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="groot_beable",
        help="Job name for logging / metadata.",
    )

    args = parser.parse_args(argv)

    if args.num_gpus <= 1:
        cmd = build_single_gpu_cmd(args)
    else:
        cmd = build_multi_gpu_cmd(args)

    print("\nLaunching training with command:\n")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining process failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()

