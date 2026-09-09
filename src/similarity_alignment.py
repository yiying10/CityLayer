#!/usr/bin/env python3
"""
Step 4 — RANSAC + Umeyama similarity transform estimation.

This is the second half of the former step3_4_pipeline.py, split out so each
step can be run and inspected independently. It consumes point_pairs.npy and
step3_filtered_matches.json written by depth_unprojection.py.

Input:
  - point_pairs.npy from Step 3            shape (N, 2, 3)
  - step3_filtered_matches.json from Step 3 (for match_vis uA/vA/uB/vB)

Outputs:
  - transform.json           (R, t, s, inlier_count, inlier_ratio, mean_residual)
  - point_pairs_inliers.npy
  - similarity_alignment.log
  - match_vis/match_step4_filtered_inliers.png (optional, when --panoA/--panoB set)

Notes:
  - Similarity is solved with outer RANSAC and inner Umeyama:
      P_B ~= s * R * P_A + t

Example:
  python3 similarity_alignment.py \
    --point-pairs /path/to/out_step3/point_pairs.npy \
    --matches /path/to/out_step3/step3_filtered_matches.json \
    --out-dir /path/to/out_step4 \
    --panoA /path/to/rgb_A.png --panoB /path/to/rgb_B.png
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np


LOG = logging.getLogger("similarity_alignment")


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(out_dir / "similarity_alignment.log")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(fh)
    LOG.addHandler(ch)


def load_matches(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("step3_filtered_matches.json must contain a list")
    return data


def umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    """Solve dst ~= s * R * src + t."""
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Umeyama requires matched point sets with at least 3 points")

    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / float(n)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    if var_src <= 1e-12:
        raise ValueError("Degenerate source point set for Umeyama")

    s = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_dst - s * (R @ mu_src)
    return R, t, s


def residuals(src: np.ndarray, dst: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    pred = (s * (R @ src.T).T) + t[None, :]
    return np.linalg.norm(pred - dst, axis=1)


def rotation_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw))


def translation_distance(t: np.ndarray) -> float:
    return float(np.linalg.norm(t))


def ransac_similarity(point_pairs: np.ndarray, iterations: int, threshold: float, min_samples: int = 4):
    if point_pairs.shape[0] < min_samples:
        raise ValueError(f"Need at least {min_samples} point pairs, got {point_pairs.shape[0]}")

    src = point_pairs[:, 0, :]
    dst = point_pairs[:, 1, :]
    n = point_pairs.shape[0]
    rng = np.random.default_rng(42)

    best = None
    best_mask = None
    best_count = -1
    best_mean = float("inf")

    for _ in range(iterations):
        idx = rng.choice(n, size=min_samples, replace=False)
        try:
            R, t, s = umeyama_similarity(src[idx], dst[idx])
        except Exception:
            continue

        err = residuals(src, dst, R, t, s)
        mask = err < threshold
        count = int(mask.sum())
        mean_err = float(err[mask].mean()) if count > 0 else float("inf")

        if count > best_count or (count == best_count and mean_err < best_mean):
            best = (R, t, s)
            best_mask = mask
            best_count = count
            best_mean = mean_err

    if best is None or best_mask is None:
        raise RuntimeError("RANSAC failed to find a valid similarity transform")

    R0, t0, s0 = best
    inlier_src = src[best_mask]
    inlier_dst = dst[best_mask]
    R, t, s = umeyama_similarity(inlier_src, inlier_dst)
    err = residuals(src, dst, R, t, s)
    final_mask = err < threshold
    return (R, t, s), final_mask, err


def draw_matches_canvas(
    pano_a_path: str,
    pano_b_path: str,
    matches_pano: List[dict[str, Any]],
    out_path: str,
    max_draw: int = 400,
) -> None:
    """Side-by-side panorama match visualization (same layout as step2/step3)."""
    img_a = cv2.imread(pano_a_path)
    img_b = cv2.imread(pano_b_path)
    if img_a is None or img_b is None:
        LOG.warning("Cannot render match_vis (image missing): %s | %s", pano_a_path, pano_b_path)
        return

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    max_vis_w = 3000
    scale = min(1.0, max_vis_w / float(w_a + w_b))
    if scale < 1.0:
        img_a = cv2.resize(img_a, (int(w_a * scale), int(h_a * scale)))
        img_b = cv2.resize(img_b, (int(w_b * scale), int(h_b * scale)))
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]

    canvas = np.zeros((max(h_a, h_b), w_a + w_b, 3), dtype=np.uint8)
    canvas[:h_a, :w_a] = img_a
    canvas[:h_b, w_a:] = img_b

    rng = np.random.default_rng(42)
    for m in matches_pano[:max_draw]:
        color = tuple(int(c) for c in rng.integers(60, 230, 3))
        pt_a = (int(m["uA"] * scale), int(m["vA"] * scale))
        pt_b = (int(m["uB"] * scale) + w_a, int(m["vB"] * scale))
        cv2.circle(canvas, pt_a, 4, color, -1)
        cv2.circle(canvas, pt_b, 4, color, -1)
        cv2.line(canvas, pt_a, pt_b, color, 1, cv2.LINE_AA)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, canvas)
    LOG.info(
        "Saved step4 match_vis -> %s (%d/%d drawn)",
        out_path,
        min(len(matches_pano), max_draw),
        len(matches_pano),
    )


def save_transform(out_path: Path, R: np.ndarray, t: np.ndarray, s: float, inlier_mask: np.ndarray, err: np.ndarray):
    inlier_count = int(inlier_mask.sum())
    total = int(inlier_mask.shape[0])
    roll_deg, pitch_deg, yaw_deg = rotation_to_euler_deg(R)
    trans_dist = translation_distance(t)

    data = {
        "R": R.astype(np.float64).tolist(),
        "t": t.astype(np.float64).tolist(),
        "s": float(s),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / max(total, 1)),
        "mean_residual": float(err[inlier_mask].mean()) if inlier_count > 0 else None,
        "median_residual": float(np.median(err[inlier_mask])) if inlier_count > 0 else None,
        "residual_threshold": float(np.max(err[inlier_mask])) if inlier_count > 0 else None,
        "euler_angles_deg": {
            "roll": roll_deg,
            "pitch": pitch_deg,
            "yaw": yaw_deg,
        },
        "translation_distance": trans_dist,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Step 4: RANSAC + Umeyama similarity transform estimation")
    p.add_argument("--point-pairs", required=True, help="Path to point_pairs.npy from Step 3")
    p.add_argument("--matches", required=True, help="Path to step3_filtered_matches.json from Step 3")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--iterations", type=int, default=500, help="RANSAC iterations")
    p.add_argument("--threshold", type=float, default=0.5, help="3D inlier threshold in A units")
    p.add_argument("--min-samples", type=int, default=4, help="RANSAC minimal sample size")
    p.add_argument(
        "--panoA",
        default=None,
        help="Optional equirect path A for match_vis PNGs (with --panoB)",
    )
    p.add_argument(
        "--panoB",
        default=None,
        help="Optional equirect path B for match_vis PNGs (with --panoA)",
    )
    p.add_argument(
        "--match-vis-max-draw",
        type=int,
        default=400,
        help="Max matches drawn per visualization image",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)

    LOG.info("Step 4 pipeline start")
    LOG.info("Point pairs: %s", args.point_pairs)
    LOG.info("Matches: %s", args.matches)

    point_pairs = np.load(args.point_pairs)
    kept_pairs = load_matches(Path(args.matches))

    if point_pairs.shape[0] < args.min_samples:
        raise RuntimeError(f"Too few valid point pairs for RANSAC: {point_pairs.shape[0]}")

    (R, t, s), inlier_mask, err = ransac_similarity(
        point_pairs,
        iterations=args.iterations,
        threshold=args.threshold,
        min_samples=args.min_samples,
    )

    LOG.info("Step 4 inliers: %d / %d", int(inlier_mask.sum()), len(inlier_mask))
    LOG.info("Mean residual: %.6f", float(err[inlier_mask].mean()) if inlier_mask.any() else float("nan"))

    inlier_pairs = point_pairs[inlier_mask]
    src_inliers = inlier_pairs[:, 0, :]
    dst_inliers = inlier_pairs[:, 1, :]
    src_range = np.linalg.norm(src_inliers, axis=1)
    dst_range = np.linalg.norm(dst_inliers, axis=1)
    LOG.info("Inlier point distance range: src [%.3f, %.3f] dst [%.3f, %.3f]",
             src_range.min(), src_range.max(), dst_range.min(), dst_range.max())

    transform_path = out_dir / "transform.json"
    save_transform(transform_path, R, t, s, inlier_mask, err)

    roll_deg, pitch_deg, yaw_deg = rotation_to_euler_deg(R)
    trans_dist = translation_distance(t)

    LOG.info("Transform: scale=%.6f R=[\n  [%.4f %.4f %.4f]\n  [%.4f %.4f %.4f]\n  [%.4f %.4f %.4f]\n] t=[%.4f %.4f %.4f]",
             s, R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2], t[0], t[1], t[2])
    LOG.info("Euler angles: roll=%.3f pitch=%.3f yaw=%.3f deg", roll_deg, pitch_deg, yaw_deg)
    LOG.info("Translation distance: %.6f", trans_dist)

    np.save(out_dir / "point_pairs_inliers.npy", inlier_pairs)

    if args.panoA and args.panoB:
        vis_dir = out_dir / "match_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        inlier_uv = [
            {
                "uA": float(kept_pairs[i]["uA"]),
                "vA": float(kept_pairs[i]["vA"]),
                "uB": float(kept_pairs[i]["uB"]),
                "vB": float(kept_pairs[i]["vB"]),
            }
            for i in range(len(kept_pairs))
            if inlier_mask[i]
        ]
        draw_matches_canvas(
            args.panoA,
            args.panoB,
            inlier_uv,
            str(vis_dir / "match_step4_filtered_inliers.png"),
            max_draw=args.match_vis_max_draw,
        )
    else:
        LOG.info("Skipping match_vis (--panoA/--panoB not both set)")

    LOG.info("Wrote transform.json, point_pairs_inliers.npy")
    LOG.info("Step 4 pipeline complete")


if __name__ == "__main__":
    main()
