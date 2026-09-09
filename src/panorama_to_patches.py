#!/usr/bin/env python3
"""
Step 1 — Equirectangular panoramas → perspective patches (or full-panorama
manifest), for cross-panorama feature matching.

This is the first half of the former step1_2_pipeline.py, split out so each
step can be run and inspected independently. Its output feeds directly into
feature_matching.py.

Modes:
  --match-full   Skip patch cutting; emit a manifest that points straight at
                 the two full panoramas (recommended, fewer error sources)
  (default)      Cut each panorama into perspective patches at --yaws and
                 emit a manifest listing every patch (useful when panoramas
                 are very wide or featureless at the equator)

Output (--out-dir):
  patches_A/, patches_B/    perspective patch PNGs + masks + *_patch_map.npy
                            (patch_map[y, x] = (u_pano, v_pano) float32)
                            — omitted entirely in --match-full mode
  patch_manifest.json       {match_full, panoA, panoB, yaws, patches_A, patches_B}
  panorama_to_patches.log

Usage:
  # Recommended: full panorama matching
  python panorama_to_patches.py \
    --panoA A.jpg --panoB B.jpg --out-dir out --match-full

  # Patch-based (useful when panoramas are very wide or featureless at equator)
  python panorama_to_patches.py \
    --panoA A.jpg --panoB B.jpg --out-dir out

  # With vehicle/inpaint masks (white=valid region):
  python panorama_to_patches.py \
    --panoA A.jpg --panoB B.jpg --maskA A_mask.png --maskB B_mask.png \
    --out-dir out --match-full
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

LOG = logging.getLogger("panorama_to_patches")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(out_dir / "panorama_to_patches.log")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(fh)
    LOG.addHandler(ch)


# ---------------------------------------------------------------------------
# Equirectangular → perspective patches
# ---------------------------------------------------------------------------

def equirect_to_patches(
    pano_path: str,
    mask_path: Optional[str],
    yaws_deg: List[float],
    fov_deg: float,
    out_res: int,
    out_dir: Path,
    prefix: str,
) -> List[Dict[str, Any]]:
    """
    Reproject an equirectangular panorama into perspective patches.

    Returns list of dicts:
        patch_path : str
        mask_path  : Optional[str]
        patch_map  : ndarray (H, W, 2)  — (u_pano, v_pano) float32
        yaw_deg    : float
    """
    img = cv2.imread(pano_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read panorama: {pano_path}")

    mask_img = None
    if mask_path:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            LOG.warning("Cannot read mask %s — ignoring", mask_path)

    pH, pW = img.shape[:2]
    W = H = out_res
    fov = np.deg2rad(fov_deg)
    fx = (W / 2.0) / np.tan(fov / 2.0)
    cx, cy = W / 2.0, H / 2.0

    # Unit direction vectors in camera space (Z forward, X right, Y up)
    xs = (np.arange(W) - cx) / fx
    ys = (np.arange(H) - cy) / fx
    gx, gy = np.meshgrid(xs, ys)
    cam = np.stack([gx, -gy, np.ones_like(gx)], axis=-1)
    cam /= np.linalg.norm(cam, axis=-1, keepdims=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for yaw_deg in yaws_deg:
        yaw = np.deg2rad(yaw_deg)
        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [           0, 1,           0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ])
        dirs = (cam.reshape(-1, 3) @ Ry.T).reshape(H, W, 3)

        # Direction → equirectangular (u, v)
        lon = np.arctan2(dirs[..., 0], dirs[..., 2])       # -π … π
        lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))  # -π/2 … π/2
        u_pano = ((lon + np.pi) / (2.0 * np.pi) * pW).astype(np.float32)
        v_pano = ((np.pi / 2.0 - lat) / np.pi * pH).astype(np.float32)

        patch = cv2.remap(img, u_pano, v_pano,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)

        patch_path = out_dir / f"{prefix}_yaw{int(yaw_deg):03d}.png"
        cv2.imwrite(str(patch_path), patch)

        pmask_path = None
        if mask_img is not None:
            pmask = cv2.remap(mask_img, u_pano, v_pano,
                              interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_WRAP)
            pmask_path = out_dir / f"{prefix}_yaw{int(yaw_deg):03d}_mask.png"
            cv2.imwrite(str(pmask_path), pmask)

        # patch_map[y, x] = (u_pano, v_pano) in original panorama pixels
        patch_map = np.stack([u_pano, v_pano], axis=-1)
        patch_map_path = str(patch_path.with_suffix("")) + "_patch_map.npy"
        np.save(patch_map_path, patch_map)

        results.append({
            "patch_path": str(patch_path),
            "mask_path":  str(pmask_path) if pmask_path else None,
            "patch_map_path": patch_map_path,
            "yaw_deg":    yaw_deg,
            "is_full_panorama": False,
        })
        LOG.info("  Patch %s saved (yaw=%.0f°)", patch_path.name, yaw_deg)

    return results


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(args) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)
    LOG.info("=== Step 1: panorama → patches ===")
    LOG.info("panoA=%s  panoB=%s  match_full=%s",
             args.panoA, args.panoB, args.match_full)

    if args.match_full:
        LOG.info("--- Full-panorama mode: skipping patch cutting ---")
        patches_A = [{
            "patch_path": args.panoA, "mask_path": args.maskA,
            "patch_map_path": None, "yaw_deg": 0.0, "is_full_panorama": True,
        }]
        patches_B = [{
            "patch_path": args.panoB, "mask_path": args.maskB,
            "patch_map_path": None, "yaw_deg": 0.0, "is_full_panorama": True,
        }]
    else:
        LOG.info("--- Patch-based mode: cutting %d yaws ---", len(args.yaws))
        patches_A = equirect_to_patches(
            args.panoA, args.maskA, args.yaws, args.fov, args.res,
            out_dir / "patches_A", prefix="A")
        patches_B = equirect_to_patches(
            args.panoB, args.maskB, args.yaws, args.fov, args.res,
            out_dir / "patches_B", prefix="B")

    manifest = {
        "match_full": bool(args.match_full),
        "panoA": args.panoA,
        "panoB": args.panoB,
        "yaws": args.yaws,
        "patches_A": patches_A,
        "patches_B": patches_B,
    }
    manifest_path = out_dir / "patch_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    LOG.info("Saved manifest (%d A patches, %d B patches) → %s",
             len(patches_A), len(patches_B), manifest_path)
    LOG.info("=== Step 1 complete ===")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 1: cut equirectangular panoramas into perspective "
                    "patches (or a full-panorama manifest) for matching")
    p.add_argument("--panoA",  required=True,  help="Equirectangular panorama A")
    p.add_argument("--panoB",  required=True,  help="Equirectangular panorama B")
    p.add_argument("--maskA",  default=None,   help="Vehicle/inpaint mask A (white=valid)")
    p.add_argument("--maskB",  default=None,   help="Vehicle/inpaint mask B (white=valid)")
    p.add_argument("--out-dir", default="output_step1", help="Output directory")

    p.add_argument("--match-full", action="store_true", default=False,
                   help="Match full panoramas directly instead of cutting patches "
                        "(recommended: fewer coordinate conversion errors)")

    # Patch-mode options (ignored when --match-full is set)
    p.add_argument("--yaws", type=float, nargs="+",
                   default=[0, 45, 90, 135, 180, 225, 270, 315])
    p.add_argument("--fov",  type=float, default=90.0)
    p.add_argument("--res",  type=int,   default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
