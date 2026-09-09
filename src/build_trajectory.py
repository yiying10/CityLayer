#!/usr/bin/env python3
import argparse, csv, json, math, numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
from plyfile import PlyData
from pathlib import Path


def ease(t):
    return t * t * (3 - 2 * t)


def yaw_to_R(yaw_deg):
    rad = math.radians(yaw_deg)
    fwd = np.array([math.sin(rad), 0, math.cos(rad)])
    up  = np.array([0., 1., 0.])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(fwd, right), fwd])


def suggest_fov_B(lat_A, lon_A, lat_B, lon_B, best_yaw, fov_near=70.0, fov_far=110.0):
    dlat = lat_B - lat_A
    dlon = lon_B - lon_A
    gps_dir = np.array([dlon, 0, dlat])
    gps_dir /= np.linalg.norm(gps_dir) + 1e-8
    rad = math.radians(best_yaw)
    cam_fwd = np.array([math.sin(rad), 0, math.cos(rad)])
    dot = float(np.dot(gps_dir, cam_fwd))
    print(f"GPS dot 相機朝向: {dot:.3f}")
    if dot > 0:
        print(f"→ 靠近場景，建議 fov_B={fov_near}°")
        return fov_near
    else:
        print(f"→ 遠離場景，建議 fov_B={fov_far}°")
        return fov_far


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_A",       required=True)
    p.add_argument("--model_B",       required=True)
    p.add_argument("--stats_csv",     required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--A_frames",      type=int,   default=240)
    p.add_argument("--B_frames",      type=int,   default=120)
    p.add_argument("--move_yaw",      type=float, default=180.0)
    p.add_argument("--move_scale",    type=float, default=6.0)
    p.add_argument("--rotate_frames", type=int,   default=180)
    p.add_argument("--gps_a_lat",     type=float, default=None)
    p.add_argument("--gps_a_lon",     type=float, default=None)
    p.add_argument("--gps_b_lat",     type=float, default=None)
    p.add_argument("--gps_b_lon",     type=float, default=None)
    p.add_argument("--fov_near",      type=float, default=70.0)
    p.add_argument("--fov_far",       type=float, default=110.0)
    args = p.parse_args()

    # 最佳切換 yaw
    best_yaw = None
    best_inliers = 0
    all_rows = []
    with open(args.stats_csv) as f:
        for row in csv.DictReader(f):
            n = int(row["inliers"])
            all_rows.append((float(row["yawA_deg"]), float(row["yawB_deg"]), n))
            if n > best_inliers:
                best_inliers = n
                best_yaw = float(row["yawA_deg"])
                best_yaw_B = float(row["yawB_deg"])

    print(f"最佳切換 yaw: A={best_yaw}° B={best_yaw_B}° ({best_inliers} inliers)")
    print("Top 5：")
    for yawA, yawB, n in sorted(all_rows, key=lambda x: -x[2])[:5]:
        print(f"  A={yawA:5.1f}° B={yawB:5.1f}°: {n}")

    # GPS 判斷 fov_B
    if all(v is not None for v in [args.gps_a_lat, args.gps_a_lon, args.gps_b_lat, args.gps_b_lon]):
        fov_B = suggest_fov_B(args.gps_a_lat, args.gps_a_lon,
                               args.gps_b_lat, args.gps_b_lon,
                               best_yaw, args.fov_near, args.fov_far)
    else:
        fov_B = args.fov_near
        print(f"GPS 未提供，預設 fov_B={fov_B}°")

    # 儲存 fov_B 供 shell 讀取
    out_dir = Path(args.output).parent
    (out_dir / "fov_B.txt").write_text(str(fov_B))
    print(f"fov_B={fov_B} → 已存入 {out_dir}/fov_B.txt")

    # 模型中心
    eA = PlyData.read(args.model_A).elements[0].data
    eB = PlyData.read(args.model_B).elements[0].data
    p_A = np.stack([eA['x'], eA['y'], eA['z']], axis=1).mean(axis=0)
    p_B = np.stack([eB['x'], eB['y'], eB['z']], axis=1).mean(axis=0)
    dist = np.linalg.norm(p_B - p_A)

    r_cam    = yaw_to_R(best_yaw)
    move_dir = yaw_to_R(args.move_yaw)[:, 2]
    p_approach = p_A + move_dir * dist * args.move_scale
    p_B_end    = p_B + move_dir * dist * args.move_scale

    rotate_n = args.rotate_frames
    move_n   = args.A_frames - rotate_n

    ts1 = ease(np.linspace(0, 1, rotate_n))
    r_rots = Slerp([0,1], Rot.from_matrix([np.eye(3), r_cam]))(ts1).as_matrix()
    seg1 = []
    for i in range(rotate_n):
        T = np.eye(4); T[:3,:3] = r_rots[i]; T[:3,3] = p_A
        seg1.append(T.tolist())

    ts2 = ease(np.linspace(0, 1, move_n))
    seg2 = []
    for i in range(move_n):
        T = np.eye(4); T[:3,:3] = r_cam
        T[:3,3] = (1-ts2[i])*p_A + ts2[i]*p_approach
        seg2.append(T.tolist())

    ts3 = ease(np.linspace(0, 1, args.B_frames))
    seg3 = []
    for i in range(args.B_frames):
        T = np.eye(4); T[:3,:3] = r_cam
        T[:3,3] = (1-ts3[i])*p_B + ts3[i]*p_B_end
        seg3.append(T.tolist())

    traj = seg1 + seg2 + seg3
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(traj, f)

    print(f"Phase1: 0°→{best_yaw}° ({rotate_n}幀)")
    print(f"Phase2: 往{args.move_yaw}°走 ({move_n}幀)")
    print(f"Phase3: B段繼續走 ({args.B_frames}幀)")
    print(f"Total: {len(traj)} frames, saved: {args.output}")


if __name__ == "__main__":
    main()
