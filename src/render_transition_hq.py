#!/usr/bin/env python3
"""
HQ transition - z-buffer blend, LightGlue-guided best switch search.
流程：
  Phase 1: 只用 A 渲染所有幀，用 LightGlue keypoint patch 找最佳切換幀
  Phase 2: 從最佳切換幀的相機位置出發，產生 B 的軌跡段
  Phase 3: 合成最終影片
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot, Slerp

try:
    import lpips; HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYERPANO_DIR = Path(os.environ.get("LAYERPANO_DIR", REPO_ROOT / "third_party" / "LayerPano3D_new"))
for p in (LAYERPANO_DIR, REPO_ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from arguments import GSParams, RENDER_BACKGROUND
from gaussian_renderer import render
from scene import LayerGaussian
from scene.cameras import MiniCam_GS


class GaussianRendererHQ:
    def __init__(self, model_path, opt):
        print(f"[HQ] Loading: {model_path}")
        self.opt = opt
        self.gaussians = LayerGaussian(opt.sh_degree)
        self.gaussians.load_ply(model_path)
        self.background = torch.tensor(RENDER_BACKGROUND, dtype=torch.float32, device="cuda")

    def render(self, pose, width, height, fovx, fovy):
        view = MiniCam_GS(pose, width, height, fovx, fovy)
        with torch.no_grad():
            out = render(view, self.gaussians, self.opt, self.background)
        rgb = np.round(out["render"].permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
        depth = (out["depth"]*(out["depth"]>0)).squeeze(0).cpu().numpy().astype(np.float32)
        return rgb, depth


def blend(rgb_A, depth_A, rgb_B, depth_B, alpha):
    alpha = alpha * alpha * (3 - 2 * alpha)
    valid_A, valid_B = depth_A > 0, depth_B > 0
    only_B  = valid_B & ~valid_A
    both    = valid_A & valid_B
    neither = ~valid_A & ~valid_B
    B_closer = both & (depth_A > depth_B)
    mid = 1.0 - abs(alpha - 0.5) * 2
    zw = 0.6 * (1.0 - mid * 0.5)
    z = rgb_A.copy().astype(np.float32)
    z[only_B]   = rgb_B[only_B].astype(np.float32)
    z[B_closer] = rgb_B[B_closer].astype(np.float32)
    z[neither]  = rgb_A[neither].astype(np.float32)*(1-alpha) + rgb_B[neither].astype(np.float32)*alpha
    cf = rgb_A.astype(np.float32)*(1-alpha) + rgb_B.astype(np.float32)*alpha
    return np.clip(z*zw + cf*(1-zw), 0, 255).astype(np.uint8)


def ease(t):
    return t * t * (3 - 2 * t)


def make_B_traj(switch_pose, model_B_path, n_frames):
    """B 段固定不動，保持切換時的相機姿態"""
    frames = []
    for _ in range(n_frames):
        frames.append(switch_pose.copy())
    return frames

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_A",             required=True)
    p.add_argument("--model_B",             required=True)
    p.add_argument("--trajectory",          required=True)
    p.add_argument("--transform",           default=None)
    p.add_argument("--output_video",        default="transition.mp4")
    p.add_argument("--width",               type=int, default=2048)
    p.add_argument("--height",              type=int, default=2048)
    p.add_argument("--fps",                 type=int, default=30)
    p.add_argument("--transition_duration", type=int, default=20)
    p.add_argument("--fov-deg",             type=float, default=90.0)
    p.add_argument("--fov-deg-B",           type=float, default=60.0,
                   help="B 場景的 fov（越小越放大，預設 60）")
    p.add_argument("--blend-z-weight",      type=float, default=0.6)
    p.add_argument("--B_frames",            type=int, default=120,
                   help="切換後 B 場景的幀數")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.trajectory) as f:
        traj_A = [np.array(p, dtype=np.float64) for p in json.load(f)]
    N = len(traj_A)
    print(f"[HQ] A trajectory: {N} frames")

    opt  = GSParams()
    fovx   = math.radians(args.fov_deg)
    fovy   = args.height * fovx / args.width
    fovx_B = math.radians(args.fov_deg_B)
    fovy_B = args.height * fovx_B / args.width
    rA   = GaussianRendererHQ(args.model_A, opt)
    rB   = GaussianRendererHQ(args.model_B, opt)

    # 載入 LightGlue 匹配點
    matches_path = Path(args.trajectory).parent / "step12" / "filtered_matches.json"
    kp_A = kp_B = None
    if matches_path.exists():
        with open(matches_path) as f:
            matches_raw = json.load(f)
        pano_w, pano_h = 1440, 720
        kp_A = np.clip(np.array([[m["uA"]/pano_w*args.width,
                                   m["vA"]/pano_h*args.height]
                                  for m in matches_raw]), 0,
                       [args.width-1, args.height-1]).astype(int)
        kp_B = np.clip(np.array([[m["uB"]/pano_w*args.width,
                                   m["vB"]/pano_h*args.height]
                                  for m in matches_raw]), 0,
                       [args.width-1, args.height-1]).astype(int)
        print(f"[HQ] Loaded {len(kp_A)} LightGlue keypoints")
    else:
        print(f"[HQ] WARNING: {matches_path} not found, using full-image MSE")

    # ── Phase 1: 渲染 A 軌跡，找最佳切換幀 ──────────────────────
    print("\n[HQ Phase 1] Rendering A trajectory, finding best switch frame...")
    frames_A = []
    errors   = []
    PATCH    = 16

    for i in tqdm(range(N)):
        pose = traj_A[i]
        rA_rgb, rA_d = rA.render(pose, args.width, args.height, fovx, fovy)
        rB_rgb, rB_d = rB.render(pose, args.width, args.height, fovx_B, fovy_B)
        frames_A.append((rA_rgb, rA_d))

        if kp_A is not None:
            diffs = []
            for (xa, ya), (xb, yb) in zip(kp_A, kp_B):
                pa = rA_rgb[max(0,ya-PATCH):ya+PATCH, max(0,xa-PATCH):xa+PATCH].astype(np.float32)
                pb = rB_rgb[max(0,yb-PATCH):yb+PATCH, max(0,xb-PATCH):xb+PATCH].astype(np.float32)
                if pa.size > 0 and pb.size > 0:
                    if pb.shape != pa.shape:
                        pb = cv2.resize(pb, (pa.shape[1], pa.shape[0]))
                    diffs.append(np.mean((pa - pb)**2))
            errors.append(np.mean(diffs) if diffs else 9999.0)
        else:
            diff = np.mean((rA_rgb.astype(np.float32) - rB_rgb.astype(np.float32))**2) / 255.0**2
            errors.append(diff * 50.0 + np.mean(np.abs(rA_d - rB_d)))

    best    = 239  # 固定在探索腳本找到的最佳切換幀
    t_start = max(0, best - args.transition_duration)
    t_end   = best  # blend 只到 best 幀，不超過
    print(f"[HQ] Best switch frame: {best}  (blend {t_start}–{t_end})")
    print(f"[HQ] Similarity score at best frame: {errors[best]:.4f}")

    # 儲存切換點比較圖
    out_dir  = os.path.dirname(args.output_video) or "."
    cmp_path = os.path.join(out_dir, f"compare_frame{best:04d}.png")
    rB_best, _ = rB.render(traj_A[best], args.width, args.height, fovx_B, fovy_B)
    canvas = np.zeros((args.height, args.width*2+8, 3), dtype=np.uint8)
    canvas[:, :args.width]      = frames_A[best][0]
    canvas[:, args.width+8:]    = rB_best
    canvas[:, args.width:args.width+8] = 255
    cv2.imwrite(cmp_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[HQ] Comparison saved: {cmp_path}")

    # ── Phase 2: 用 trajectory.json 的 B 段（幀 240 之後）渲染 ──
    print(f"\n[HQ Phase 2] Rendering B segment from trajectory...")
    traj_B_poses = traj_A[best:]  # best 之後的幀當 B 段
    frames_B = []
    for pose in tqdm(traj_B_poses):
        rgb, d = rB.render(pose, args.width, args.height, fovx_B, fovy_B)
        frames_B.append((rgb, d))
    print(f"[HQ] B segment: {len(frames_B)} frames")

    print("[HQ Phase 4] Composing final video...")
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))

    # A 段（0 ~ t_start）
    for i in range(t_start):
        writer.write(cv2.cvtColor(frames_A[i][0], cv2.COLOR_RGB2BGR))

    # Blend 段（t_start ~ t_end）
    blend_B_start = rB.render(traj_A[t_start], args.width, args.height, fovx_B, fovy_B)
    blend_B_end   = rB.render(traj_A[t_end],   args.width, args.height, fovx, fovy)
    # blend 段的 B 直接用 frames_B[0]（固定位置，不重新渲染）
    rB_fixed_rgb, rB_fixed_d = frames_B[0]
    for i in tqdm(range(t_start, t_end+1), desc="Blending"):
        alpha = (i - t_start) / (t_end - t_start + 1e-5)
        frame = blend(frames_A[i][0], frames_A[i][1], rB_fixed_rgb, rB_fixed_d, alpha)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # B 段
    for i, (rgb, d) in enumerate(tqdm(frames_B, desc="B segment")):
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    writer.release()
    total = t_start + (t_end - t_start + 1) + len(frames_B)
    print(f"[HQ] Saved: {args.output_video}")
    print(f"[HQ] A段: {t_start}幀 ({t_start/args.fps:.1f}s) | Blend: {t_end-t_start+1}幀 | B段: {len(frames_B)}幀 ({len(frames_B)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()