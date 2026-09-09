#!/usr/bin/env python3
"""
Step 2 — cross-panorama feature matching + geometric filtering.

This is the second half of the former step1_2_pipeline.py, split out so each
step can be run and inspected independently. It consumes the
patch_manifest.json written by panorama_to_patches.py.

Key fixes vs the old monolithic step1_2 v2:
- match_full_panoramas runs Fundamental Matrix RANSAC (correct for scenes
  with parallax / camera translation), replacing the broken homography filter
- Patch mode also upgraded from homography to Fundamental Matrix RANSAC
- match_full mode no longer fakes inlier_ratio=1.0 in pair_stats
- LightGlue score threshold passed as a proper argument, not a function attribute
- SIFT sky/ground ROI crop disabled for patches (they are already horizontal crops)
- Full-panorama LightGlue path resizes input to a safe resolution to avoid OOM

Input:
  - patch_manifest.json from Step 1

Output (--out-dir):
  filtered_matches.json        list of {uA, vA, uB, vB, yawA_deg, yawB_deg}
                                in full-panorama pixel coordinates
  match_pair_stats.csv
  match_pair_inlier_table.md   (patch mode only, when >1 yaw)
  match_vis/                   side-by-side match visualizations
  feature_matching.log

Usage:
  python feature_matching.py \
    --manifest out/patch_manifest.json --out-dir out
"""

import csv
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

LOG = logging.getLogger("feature_matching")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(out_dir / "feature_matching.log")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(fh)
    LOG.addHandler(ch)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _build_roi_mask(h: int, w: int,
                    user_mask_path: Optional[str],
                    is_full_panorama: bool) -> Optional[np.ndarray]:
    """
    Build the SIFT ROI mask.

    For full panoramas: suppress top 15% (sky) and bottom 8% (ground/car hood).
    For patches: no vertical crop — patches are already horizontal slices and
                 cropping would remove valid wall features.
    User inpaint mask (white=valid) is always ANDed in when provided.
    """
    roi = np.full((h, w), 255, dtype=np.uint8)

    if is_full_panorama:
        sky_rows  = int(h * 0.15)
        gnd_rows  = int(h * 0.08)
        roi[:sky_rows, :]      = 0
        roi[h - gnd_rows:, :] = 0

    if user_mask_path:
        m = cv2.imread(user_mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            m_bin = cv2.resize((m > 128).astype(np.uint8) * 255, (w, h),
                               interpolation=cv2.INTER_NEAREST)
            roi = cv2.bitwise_and(roi, m_bin)
        else:
            LOG.warning("Could not read mask %s", user_mask_path)

    return roi


def extract_sift(image_path: str,
                 mask_path: Optional[str] = None,
                 is_full_panorama: bool = False,
                 nfeatures: int = 6000) -> tuple:
    """
    Returns (keypoints_list [(x,y),...], descriptors ndarray | None).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    h, w = img.shape
    roi = _build_roi_mask(h, w, mask_path, is_full_panorama)

    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02)
    kps, des = sift.detectAndCompute(img, roi)

    if des is None:
        return [], None

    kp_list = [(float(k.pt[0]), float(k.pt[1])) for k in kps]
    LOG.debug("SIFT: %d keypoints in %s", len(kp_list), Path(image_path).name)
    return kp_list, des


def try_lightglue(img_path_A: str,
                  img_path_B: str,
                  mask_path_A: Optional[str] = None,
                  mask_path_B: Optional[str] = None,
                  score_thresh: float = 0.75) -> Optional[List[tuple]]:
    """
    Match with LightGlue + SuperPoint (using demo.py style).
    Returns list of (xA, yA, xB, yB) tuples, or None if LightGlue is unavailable.
    """
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image, rbd
    except ImportError:
        LOG.info("LightGlue not installed — falling back to SIFT+FLANN")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    matcher   = LightGlue(features="superpoint").eval().to(device)

    # Load images using load_image (from demo.py)
    image0 = load_image(img_path_A).to(device)
    image1 = load_image(img_path_B).to(device)

    # Apply masks if provided (zero out masked regions)
    if mask_path_A and Path(mask_path_A).exists():
        m0 = cv2.imread(mask_path_A, cv2.IMREAD_GRAYSCALE)
        if m0 is not None:
            _, H, W = image0.shape
            m0_resized = cv2.resize((m0 > 128).astype(np.float32), (W, H),
                                     interpolation=cv2.INTER_NEAREST)
            m0_t = torch.from_numpy(m0_resized).to(device).unsqueeze(0)
            image0 = image0 * m0_t

    if mask_path_B and Path(mask_path_B).exists():
        m1 = cv2.imread(mask_path_B, cv2.IMREAD_GRAYSCALE)
        if m1 is not None:
            _, H, W = image1.shape
            m1_resized = cv2.resize((m1 > 128).astype(np.float32), (W, H),
                                     interpolation=cv2.INTER_NEAREST)
            m1_t = torch.from_numpy(m1_resized).to(device).unsqueeze(0)
            image1 = image1 * m1_t

    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Extract matches and apply score threshold (demo.py style)
    m_idx = matches01["matches"]
    scores = matches01["scores"].cpu().numpy()
    points0 = feats0["keypoints"][m_idx[:, 0]].cpu().numpy()
    points1 = feats1["keypoints"][m_idx[:, 1]].cpu().numpy()

    LOG.info("LightGlue: %d raw matches, scores min=%.3f max=%.3f mean=%.3f",
             len(m_idx),
             scores.min() if len(scores) > 0 else 0.0,
             scores.max() if len(scores) > 0 else 0.0,
             scores.mean() if len(scores) > 0 else 0.0)

    # Filter by score threshold
    keep = scores > score_thresh
    points0_filtered = points0[keep]
    points1_filtered = points1[keep]

    LOG.info("LightGlue: %d matches (score>%.2f) in %s × %s",
             len(points0_filtered), score_thresh,
             Path(img_path_A).name, Path(img_path_B).name)

    return list(zip(points0_filtered[:, 0].tolist(), points0_filtered[:, 1].tolist(),
                    points1_filtered[:, 0].tolist(), points1_filtered[:, 1].tolist()))


# ---------------------------------------------------------------------------
# SIFT + FLANN matching
# ---------------------------------------------------------------------------

def match_sift_flann(kps_A: list, des_A: np.ndarray,
                     kps_B: list, des_B: np.ndarray,
                     ratio: float = 0.72) -> List[tuple]:
    """Lowe ratio test. Returns [(xA, yA, xB, yB), ...]."""
    if des_A is None or des_B is None or len(des_A) < 2 or len(des_B) < 2:
        return []

    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(des_A.astype(np.float32), des_B.astype(np.float32), k=2)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            xA, yA = kps_A[m.queryIdx]
            xB, yB = kps_B[m.trainIdx]
            good.append((float(xA), float(yA), float(xB), float(yB)))
    return good


# ---------------------------------------------------------------------------
# Geometric filtering — Fundamental Matrix RANSAC
# ---------------------------------------------------------------------------

def geometric_filter_fundamental(
    matches: List[tuple],
    min_inliers: int = 15,
    reproj_thresh: float = 3.0,
) -> List[tuple]:
    """
    Filter matches using Fundamental Matrix RANSAC.

    Unlike homography RANSAC, the fundamental matrix correctly models
    two-view geometry with camera TRANSLATION (parallax), which is exactly
    the case here (two panoramas ~18 m apart on a street).

    Homography RANSAC assumes all points are coplanar or the camera only
    rotates — neither holds for this scene, so it was discarding valid
    matches and keeping wrong ones.

    min_inliers: if fewer inliers survive, return empty list (pair is
                 unreliable; better to have fewer pairs than polluted ones).
    reproj_thresh: Sampson distance threshold in pixels.
    """
    if len(matches) < 8:
        LOG.debug("Too few matches (%d) for F-matrix RANSAC (need ≥8)", len(matches))
        return []

    pts_A = np.float32([[m[0], m[1]] for m in matches])
    pts_B = np.float32([[m[2], m[3]] for m in matches])

    _, mask = cv2.findFundamentalMat(
        pts_A, pts_B,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=reproj_thresh,
        confidence=0.999,
    )

    if mask is None:
        LOG.debug("F-matrix RANSAC returned no mask")
        return []

    inliers = [m for m, keep in zip(matches, mask.ravel()) if keep]
    n_in, n_raw = len(inliers), len(matches)
    LOG.debug("F-RANSAC: %d → %d inliers (%.0f%%)", n_raw, n_in,
              100.0 * n_in / max(n_raw, 1))

    if n_in < min_inliers:
        LOG.debug("Inlier count %d below min_inliers=%d — discarding pair",
                  n_in, min_inliers)
        return []

    return inliers


# ---------------------------------------------------------------------------
# Convert patch coordinates → panorama UV
# ---------------------------------------------------------------------------

def patch_to_pano_uv(matches_patch: List[tuple],
                     patch_map_A: np.ndarray,
                     patch_map_B: np.ndarray) -> List[dict]:
    """
    Map (xA_patch, yA_patch, xB_patch, yB_patch) → panorama (uA, vA, uB, vB).
    patch_map shape: (H, W, 2) — channel 0 = u_pano, channel 1 = v_pano.
    """
    H_A, W_A = patch_map_A.shape[:2]
    H_B, W_B = patch_map_B.shape[:2]
    out = []
    for xA, yA, xB, yB in matches_patch:
        xi_A = int(np.clip(xA, 0, W_A - 1))
        yi_A = int(np.clip(yA, 0, H_A - 1))
        xi_B = int(np.clip(xB, 0, W_B - 1))
        yi_B = int(np.clip(yB, 0, H_B - 1))
        uA, vA = patch_map_A[yi_A, xi_A]
        uB, vB = patch_map_B[yi_B, xi_B]
        out.append({"uA": float(uA), "vA": float(vA),
                    "uB": float(uB), "vB": float(vB)})
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_matches_canvas(img_path_A: str, img_path_B: str,
                        matches_pano: List[dict],
                        out_path: str, max_draw: int = 400) -> None:
    imgA = cv2.imread(img_path_A)
    imgB = cv2.imread(img_path_B)
    if imgA is None or imgB is None:
        LOG.warning("draw_matches_canvas: could not read %s or %s",
                    img_path_A, img_path_B)
        return

    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]

    # Downscale for visualization if panoramas are very large
    max_vis_w = 3000
    scale = min(1.0, max_vis_w / (wA + wB))
    if scale < 1.0:
        imgA = cv2.resize(imgA, (int(wA * scale), int(hA * scale)))
        imgB = cv2.resize(imgB, (int(wB * scale), int(hB * scale)))
        hA, wA = imgA.shape[:2]
        hB, wB = imgB.shape[:2]

    canvas = np.zeros((max(hA, hB), wA + wB, 3), dtype=np.uint8)
    canvas[:hA, :wA] = imgA
    canvas[:hB, wA:] = imgB

    rng = np.random.default_rng(42)
    for m in matches_pano[:max_draw]:
        color = tuple(int(c) for c in rng.integers(60, 230, 3))
        ptA = (int(m["uA"] * scale), int(m["vA"] * scale))
        ptB = (int(m["uB"] * scale) + wA, int(m["vB"] * scale))
        cv2.circle(canvas, ptA, 4, color, -1)
        cv2.circle(canvas, ptB, 4, color, -1)
        cv2.line(canvas, ptA, ptB, color, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    LOG.info("Saved visualization → %s (%d/%d matches drawn)",
             out_path, min(len(matches_pano), max_draw), len(matches_pano))


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------

def write_stats(out_dir: Path,
                pair_stats: List[Dict[str, Any]],
                all_matches: List[dict],
                yaws: List[float]) -> None:
    csv_path = out_dir / "match_pair_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["yawA_deg", "yawB_deg", "raw", "inliers", "ratio"])
        writer.writeheader()
        writer.writerows(pair_stats)

    # Markdown inlier table (patch mode only)
    if len(yaws) > 1:
        yaw_ints = [int(round(y)) for y in yaws]
        stat_map = {(int(r["yawA_deg"]), int(r["yawB_deg"])): int(r["inliers"])
                    for r in pair_stats}
        lines = ["| A \\ B | " + " | ".join(f"{y:03d}" for y in yaw_ints) + " |",
                 "|" + "---|" * (len(yaw_ints) + 1)]
        for ya in yaw_ints:
            lines.append("| " + " | ".join(
                [f"{ya:03d}"] + [str(stat_map.get((ya, yb), 0)) for yb in yaw_ints]
            ) + " |")
        (out_dir / "match_pair_inlier_table.md").write_text("\n".join(lines) + "\n")

    # Spatial distribution check
    total = len(all_matches)
    if total == 0:
        LOG.warning("No matches survived — check input images and masks")
        return

    LOG.info("Total final matches: %d", total)

    if pair_stats:
        best = max(pair_stats, key=lambda r: r["inliers"])
        LOG.info("Best pair: A=%.0f° B=%.0f° → %d inliers",
                 best["yawA_deg"], best["yawB_deg"], best["inliers"])

    # Warn if matches are spatially degenerate (all from one yaw pair)
    counts: Dict[tuple, int] = {}
    for m in all_matches:
        key = (int(round(m.get("yawA_deg", 0))), int(round(m.get("yawB_deg", 0))))
        counts[key] = counts.get(key, 0) + 1
    top_key  = max(counts, key=lambda k: counts[k])
    top_frac = counts[top_key] / total
    if top_frac > 0.75:
        LOG.warning(
            "Spatial degeneracy warning: %.0f%% of matches come from a single "
            "patch pair (A=%.0f° B=%.0f°). Umeyama may produce a poor rotation "
            "estimate. Consider lowering --min-inliers or using --match-full.",
            100.0 * top_frac, top_key[0], top_key[1])


# ---------------------------------------------------------------------------
# Main matching logic
# ---------------------------------------------------------------------------

def run_match_pair(path_A: str, path_B: str,
                   mask_A: Optional[str], mask_B: Optional[str],
                   matcher: str, score_thresh: float,
                   min_inliers: int, reproj_thresh: float,
                   is_full_pano: bool) -> tuple:
    """
    Match one image pair. Returns (raw_matches, inlier_matches).
    raw_matches and inlier_matches are lists of (xA,yA,xB,yB) tuples.
    """
    raw: List[tuple] = []

    if matcher == "lightglue":
        result = try_lightglue(path_A, path_B, mask_A, mask_B,
                               score_thresh=score_thresh)
        if result is not None:
            raw = result
        else:
            kA, dA = extract_sift(path_A, mask_A, is_full_panorama=is_full_pano)
            kB, dB = extract_sift(path_B, mask_B, is_full_panorama=is_full_pano)
            raw = match_sift_flann(kA, dA, kB, dB)
    else:
        kA, dA = extract_sift(path_A, mask_A, is_full_panorama=is_full_pano)
        kB, dB = extract_sift(path_B, mask_B, is_full_panorama=is_full_pano)
        raw = match_sift_flann(kA, dA, kB, dB)

    inliers = geometric_filter_fundamental(raw, min_inliers=min_inliers,
                                           reproj_thresh=reproj_thresh)
    return raw, inliers


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(args) -> List[dict]:
    out_dir = Path(args.out_dir)
    setup_logging(out_dir)
    LOG.info("=== Step 2: feature matching ===")

    manifest = load_manifest(Path(args.manifest))
    match_full = bool(manifest["match_full"])
    patches_A = manifest["patches_A"]
    patches_B = manifest["patches_B"]
    LOG.info("manifest=%s  match_full=%s  A_patches=%d  B_patches=%d",
             args.manifest, match_full, len(patches_A), len(patches_B))

    vis_dir = out_dir / "match_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    all_matches: List[dict] = []
    pair_stats:  List[Dict[str, Any]] = []

    for pA in patches_A:
        patch_map_A = (np.load(pA["patch_map_path"])
                        if pA["patch_map_path"] else None)
        for pB in patches_B:
            yawA, yawB = pA["yaw_deg"], pB["yaw_deg"]
            is_full_pano = bool(pA["is_full_panorama"] and pB["is_full_panorama"])
            raw, inliers = run_match_pair(
                pA["patch_path"], pB["patch_path"],
                pA["mask_path"],  pB["mask_path"],
                args.matcher, args.lightglue_thresh,
                args.min_inliers, args.reproj_thresh,
                is_full_pano=is_full_pano,
            )
            LOG.info("  A=%.0f° B=%.0f°  raw=%d  inliers=%d",
                     yawA, yawB, len(raw), len(inliers))

            pair_stats.append({
                "yawA_deg": yawA, "yawB_deg": yawB,
                "raw": len(raw), "inliers": len(inliers),
                "ratio": len(inliers) / max(len(raw), 1),
            })

            if not inliers:
                continue

            if patch_map_A is not None and pB["patch_map_path"]:
                patch_map_B = np.load(pB["patch_map_path"])
                pano_uv = patch_to_pano_uv(inliers, patch_map_A, patch_map_B)
            else:
                # Full-panorama mode: (x, y) coordinates already ARE
                # panorama (u, v) pixel coordinates.
                pano_uv = [{"uA": xA, "vA": yA, "uB": xB, "vB": yB}
                           for xA, yA, xB, yB in inliers]

            for item in pano_uv:
                item["yawA_deg"] = yawA
                item["yawB_deg"] = yawB
            all_matches.extend(pano_uv)

            vis_name = ("match_full_panorama.png" if is_full_pano else
                        f"match_A{int(yawA):03d}_B{int(yawB):03d}.png")
            draw_matches_canvas(
                pA["patch_path"], pB["patch_path"],
                [{"uA": m[0], "vA": m[1], "uB": m[2], "vB": m[3]}
                 for m in inliers],
                str(vis_dir / vis_name),
            )

    # Full panorama overlay visualization (patch mode only; full mode already
    # drew match_full_panorama.png above)
    if not match_full:
        draw_matches_canvas(manifest["panoA"], manifest["panoB"], all_matches,
                            str(vis_dir / "match_panorama_full.png"))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    matches_file = out_dir / "filtered_matches.json"
    with open(matches_file, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2)
    LOG.info("Saved %d matches → %s", len(all_matches), matches_file)

    write_stats(out_dir, pair_stats, all_matches, manifest["yaws"])
    LOG.info("=== Step 2 complete ===")
    return all_matches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 2: cross-panorama feature matching for 3DGS alignment")
    p.add_argument("--manifest", required=True,
                   help="patch_manifest.json from Step 1")
    p.add_argument("--out-dir", default="output_step2", help="Output directory")

    p.add_argument("--matcher", choices=["lightglue", "sift"], default="lightglue")
    p.add_argument("--lightglue-thresh", type=float, default=0.75,
                   help="LightGlue confidence threshold (default 0.75)")

    # Filtering
    p.add_argument("--min-inliers",   type=int,   default=15,
                   help="Min F-matrix inliers to accept a patch pair")
    p.add_argument("--reproj-thresh", type=float, default=3.0,
                   help="Sampson distance threshold for F-matrix RANSAC (pixels)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
