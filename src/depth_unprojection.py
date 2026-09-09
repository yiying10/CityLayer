#!/usr/bin/env python3
"""
Step 3 — depth-based match filtering + unprojection to 3D points.

This is the first half of the former step3_4_pipeline.py, split out so each
step can be run and inspected independently. Its output feeds directly into
similarity_alignment.py.

Input:
  - filtered_matches.json from Step 2
  - depth_A.npy / depth_B.npy

Outputs:
  - point_pairs.npy           shape (N, 2, 3)
  - step3_filtered_matches.json
  - depth_unprojection.log
  - match_vis/match_step3_depth_filtered.png (optional, when --panoA/--panoB set)

Notes:
  - Uses the panorama UV convention from feature_matching.py:
      lon = u / W * 2pi - pi
      lat = pi/2 - v / H * pi
      dir = [cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon)]
  - Depth reliability filtering follows the requested rules:
      depth > 0.1
      depth < 95th percentile of valid depth values
      local 3x3 std < 0.15 * median depth

Example:
  python3 depth_unprojection.py \
    --matches /path/to/out/filtered_matches.json \
    --depthA /path/to/depth_A.npy \
    --depthB /path/to/depth_B.npy \
    --out-dir /path/to/out_step3 \
    --panoA /path/to/rgb_A.png --panoB /path/to/rgb_B.png
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np


LOG = logging.getLogger("depth_unprojection")


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(out_dir / "depth_unprojection.log")
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
        raise ValueError("filtered_matches.json must contain a list")
    return data


def load_depth(path: Path) -> np.ndarray:
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D, got shape {depth.shape} from {path}")
    return depth.astype(np.float64)


def uv_to_dir(u: float, v: float, width: int, height: int) -> np.ndarray:
    lon = (u / float(width)) * (2.0 * math.pi) - math.pi
    lat = (math.pi / 2.0) - (v / float(height)) * math.pi
    cos_lat = math.cos(lat)
    return np.array([
        cos_lat * math.sin(lon),
        math.sin(lat),
        cos_lat * math.cos(lon),
    ], dtype=np.float64)


def extract_window(depth: np.ndarray, u: int, v: int) -> np.ndarray:
    h, w = depth.shape
    rows = np.clip(np.array([v - 1, v, v + 1]), 0, h - 1)
    cols = np.array([(u - 1) % w, u % w, (u + 1) % w])
    return depth[np.ix_(rows, cols)]


def depth_percentile_95(depth: np.ndarray) -> float:
    valid = depth[depth > 0.1]
    if valid.size == 0:
        raise ValueError("Depth map has no valid positive values")
    return float(np.percentile(valid, 95))


def filter_and_unproject(matches, depth_a, depth_b):
    h_a, w_a = depth_a.shape
    h_b, w_b = depth_b.shape
    p95_a = depth_percentile_95(depth_a)
    p95_b = depth_percentile_95(depth_b)

    kept_pairs = []
    report = {
        "total_matches": len(matches),
        "kept_matches": 0,
        "rejected_depth_low": 0,
        "rejected_depth_high": 0,
        "rejected_depth_std": 0,
        "rejected_missing": 0,
    }

    for item in matches:
        try:
            u_a = float(item["uA"])
            v_a = float(item["vA"])
            u_b = float(item["uB"])
            v_b = float(item["vB"])
        except Exception:
            report["rejected_missing"] += 1
            continue

        ua = int(np.clip(round(u_a), 0, w_a - 1))
        va = int(np.clip(round(v_a), 0, h_a - 1))
        ub = int(np.clip(round(u_b), 0, w_b - 1))
        vb = int(np.clip(round(v_b), 0, h_b - 1))

        win_a = extract_window(depth_a, ua, va)
        win_b = extract_window(depth_b, ub, vb)

        med_a = float(np.median(win_a))
        med_b = float(np.median(win_b))
        std_a = float(np.std(win_a))
        std_b = float(np.std(win_b))

        if med_a <= 0.1 or med_b <= 0.1:
            report["rejected_depth_low"] += 1
            continue
        if med_a >= p95_a or med_b >= p95_b:
            report["rejected_depth_high"] += 1
            continue
        if std_a >= 0.15 * med_a or std_b >= 0.15 * med_b:
            report["rejected_depth_std"] += 1
            continue

        dir_a = uv_to_dir(u_a, v_a, w_a, h_a)
        dir_b = uv_to_dir(u_b, v_b, w_b, h_b)
        p_a = dir_a * med_a
        p_b = dir_b * med_b

        kept_pairs.append({
            "uA": u_a,
            "vA": v_a,
            "uB": u_b,
            "vB": v_b,
            "depthA": med_a,
            "depthB": med_b,
            "P_A": p_a.tolist(),
            "P_B": p_b.tolist(),
            "yawA_deg": item.get("yawA_deg"),
            "yawB_deg": item.get("yawB_deg"),
        })

    report["kept_matches"] = len(kept_pairs)
    if kept_pairs:
        pairs = np.asarray([[p["P_A"], p["P_B"]] for p in kept_pairs], dtype=np.float64)
        print("P_A range:", pairs[:, 0].min(axis=0), pairs[:, 0].max(axis=0))
        print("P_B range:", pairs[:, 1].min(axis=0), pairs[:, 1].max(axis=0))
        print("depth_A shape:", depth_a.shape, "pano UV max:",
              max(m["uA"] for m in matches), max(m["vA"] for m in matches))
    return kept_pairs, report


def draw_matches_canvas(
    pano_a_path: str,
    pano_b_path: str,
    matches_pano: List[dict[str, Any]],
    out_path: str,
    max_draw: int = 400,
) -> None:
    """Side-by-side panorama match visualization (same layout as step2)."""
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
        "Saved step3 match_vis -> %s (%d/%d drawn)",
        out_path,
        min(len(matches_pano), max_draw),
        len(matches_pano),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Step 3: depth-filter matches and unproject to 3D")
    p.add_argument("--matches", required=True, help="Path to filtered_matches.json from Step 2")
    p.add_argument("--depthA", required=True, help="Path to depth_A.npy")
    p.add_argument("--depthB", required=True, help="Path to depth_B.npy")
    p.add_argument("--out-dir", required=True, help="Output directory")
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

    LOG.info("Step 3 pipeline start")
    LOG.info("Matches: %s", args.matches)
    LOG.info("Depth A: %s", args.depthA)
    LOG.info("Depth B: %s", args.depthB)

    matches = load_matches(Path(args.matches))
    depth_a = load_depth(Path(args.depthA))
    depth_b = load_depth(Path(args.depthB))

    kept_pairs, report = filter_and_unproject(matches, depth_a, depth_b)
    LOG.info("Step 3 kept %d / %d matches", report["kept_matches"], report["total_matches"])
    LOG.info("Rejected: low=%d high=%d std=%d missing=%d",
             report["rejected_depth_low"], report["rejected_depth_high"],
             report["rejected_depth_std"], report["rejected_missing"])

    if kept_pairs:
        depths_a = [item["depthA"] for item in kept_pairs]
        depths_b = [item["depthB"] for item in kept_pairs]
        LOG.info("Depth A range: [%.3f, %.3f] mean=%.3f", min(depths_a), max(depths_a), np.mean(depths_a))
        LOG.info("Depth B range: [%.3f, %.3f] mean=%.3f", min(depths_b), max(depths_b), np.mean(depths_b))
        depth_ratio = np.mean(depths_a) / np.mean(depths_b) if np.mean(depths_b) > 0 else 1.0
        LOG.info("Mean depth ratio (A/B): %.3f", depth_ratio)

    if len(kept_pairs) < 3:
        raise RuntimeError(f"Too few valid point pairs after depth filtering: {len(kept_pairs)}")

    point_pairs = np.asarray([
        [item["P_A"], item["P_B"]]
        for item in kept_pairs
    ], dtype=np.float64)
    np.save(out_dir / "point_pairs.npy", point_pairs)

    filtered_matches_path = out_dir / "step3_filtered_matches.json"
    with open(filtered_matches_path, "w", encoding="utf-8") as f:
        json.dump(kept_pairs, f, indent=2)

    if args.panoA and args.panoB:
        vis_dir = out_dir / "match_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        depth_only = [
            {"uA": float(p["uA"]), "vA": float(p["vA"]), "uB": float(p["uB"]), "vB": float(p["vB"])}
            for p in kept_pairs
        ]
        draw_matches_canvas(
            args.panoA,
            args.panoB,
            depth_only,
            str(vis_dir / "match_step3_depth_filtered.png"),
            max_draw=args.match_vis_max_draw,
        )
    else:
        LOG.info("Skipping match_vis (--panoA/--panoB not both set)")

    LOG.info("Wrote point_pairs.npy, step3_filtered_matches.json")
    LOG.info("Step 3 pipeline complete")


if __name__ == "__main__":
    main()
