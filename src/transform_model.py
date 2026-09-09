#!/usr/bin/env python3
"""
Step 5: apply the Step 4 similarity transform to model_B.ply and export
model_B_aligned.ply in model_A coordinates.

Input:
  - model_B.ply (3DGS Gaussian PLY)
  - transform.json from Step 3-4

Output:
  - model_B_aligned.ply
  - transform_model.log

Supported PLY layouts:
  - Standard 3DGS Gaussian PLY with fields:
      x, y, z, f_dc_*, f_rest_*, opacity, scale_*, rot_*
  - Generic point-cloud PLY with at least x, y, z (other fields preserved)

Transform convention from Step 4:
  P_B ~= s * R * P_A + t
Therefore:
  P_A = (1 / s) * R^T * (P_B - t)
    q_A = q_B ⊗ q(R^T)
    scale_A(log-space) = scale_B - log(s)

Quaternion convention in this project is [w, x, y, z].
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


LOG = logging.getLogger("transform_model")


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(out_dir / "transform_model.log")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    LOG.setLevel(logging.DEBUG)
    LOG.addHandler(fh)
    LOG.addHandler(ch)


def load_transform(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    R = np.asarray(data["R"], dtype=np.float64)
    t = np.asarray(data["t"], dtype=np.float64)
    s = float(data["s"])
    if R.shape != (3, 3):
        raise ValueError(f"Expected R shape (3, 3), got {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"Expected t shape (3,), got {t.shape}")
    if not np.isfinite(s) or abs(s) < 1e-12:
        raise ValueError(f"Invalid scale in transform.json: {s}")
    return R, t, s, data


def matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized [w, x, y, z] quaternion."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        diag = np.diag(R)
        if diag[0] > diag[1] and diag[0] > diag[2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif diag[1] > diag[2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Degenerate quaternion from rotation matrix")
    return q / norm


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in [w, x, y, z] order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def align_ply(model_b_path: Path, transform_path: Path, out_path: Path):
    R, t, s, meta = load_transform(transform_path)
    R_inv = R.T
    q_inv = matrix_to_quaternion_wxyz(R_inv)

    plydata = PlyData.read(str(model_b_path))
    if len(plydata.elements) == 0 or plydata.elements[0].name != "vertex":
        raise ValueError("PLY file does not contain a vertex element")

    vertex = plydata.elements[0]
    data = np.array(vertex.data, copy=True)
    props = vertex.properties

    if not all(name in data.dtype.names for name in ("x", "y", "z")):
        raise ValueError("PLY vertex element must contain x/y/z fields")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    xyz_aligned = ((xyz - t[None, :]) @ R_inv.T) / s
    data["x"] = xyz_aligned[:, 0].astype(data["x"].dtype, copy=False)
    data["y"] = xyz_aligned[:, 1].astype(data["y"].dtype, copy=False)
    data["z"] = xyz_aligned[:, 2].astype(data["z"].dtype, copy=False)

    # In 3DGS PLY, scale_* stores log-scale parameters, not linear scale.
    # For inverse similarity with linear factor (1/s), we add log(1/s) = -log(s).
    scale_names = sorted([name for name in data.dtype.names if name.startswith("scale_")], key=lambda n: int(n.split("_")[-1]))
    if scale_names:
        log_inv_s = -np.log(s)
        for name in scale_names:
            data[name] = (np.asarray(data[name], dtype=np.float64) + log_inv_s).astype(data[name].dtype, copy=False)

    rot_names = sorted([name for name in data.dtype.names if name.startswith("rot_")], key=lambda n: int(n.split("_")[-1]))
    if len(rot_names) == 4:
        rots = np.stack([np.asarray(data[name], dtype=np.float64) for name in rot_names], axis=1)
        # Keep quaternion composition order consistent with align_3dgs.py:
        # q_new = q_old ⊗ q_transform.
        rots_aligned = np.vstack([quat_multiply_wxyz(q, q_inv) for q in rots])
        rots_norm = np.linalg.norm(rots_aligned, axis=1, keepdims=True)
        rots_aligned = rots_aligned / np.clip(rots_norm, 1e-12, None)
        for idx, name in enumerate(rot_names):
            data[name] = rots_aligned[:, idx].astype(data[name].dtype, copy=False)
    elif len(rot_names) not in (0, 4):
        LOG.warning("Found %d rotation fields; expected 0 or 4. Rotation will be left unchanged.", len(rot_names))

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=plydata.text).write(str(out_path))

    LOG.info("Loaded transform: inlier_count=%s inlier_ratio=%s mean_residual=%s", meta.get("inlier_count"), meta.get("inlier_ratio"), meta.get("mean_residual"))
    LOG.info("Wrote aligned model to %s", out_path)
    LOG.info("Applied xyz, scale and rotation transforms using q_inv=[w,x,y,z]=%s", q_inv.tolist())


def parse_args():
    p = argparse.ArgumentParser(description="Step 5: apply similarity transform to a Gaussian PLY")
    p.add_argument("--modelB", required=True, help="Input model_B.ply")
    p.add_argument("--transform", required=True, help="transform.json from Step 4")
    p.add_argument("--out", required=True, help="Output model_B_aligned.ply")
    p.add_argument("--out-dir", default=None, help="Optional log/output directory (defaults to output file directory)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    log_dir = Path(args.out_dir) if args.out_dir else out_path.parent
    setup_logging(log_dir)

    LOG.info("Step 5 start")
    LOG.info("Input model: %s", args.modelB)
    LOG.info("Transform: %s", args.transform)
    LOG.info("Output: %s", args.out)

    align_ply(Path(args.modelB), Path(args.transform), out_path)

    LOG.info("Step 5 complete")


if __name__ == "__main__":
    main()