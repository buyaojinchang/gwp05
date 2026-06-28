"""Calibration probe for the SONIC decoder obs assembly.

Feeds the *ground-truth* motion token + teacher-forced proprioception (from the
raw pick_place dataset) through model_decoder.onnx and compares the decoded body
joints against ground-truth ``action.wbc``. With the GT token, a correct obs
assembly should reproduce action.wbc closely -- this lets us calibrate joint
order / default-pose offset / history direction / angular-velocity *without* the
world model.
"""
import argparse
import itertools
import os

import numpy as np
import onnxruntime as ort
import pandas as pd

# ── G1 joint metadata (from gear_sonic/envs/manager_env/robots/g1.py) ─────────
G1_ISAACLAB_TO_MUJOCO_DOF = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                             11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
G1_MUJOCO_TO_ISAACLAB_DOF = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                             16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
ISAACLAB_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def action_scale_isaaclab():
    """0.25 * effort_limit / stiffness per joint (G1_MODEL_12_ACTION_SCALE)."""
    nf2 = (10 * 2 * np.pi) ** 2
    S_5020 = 0.003609725 * nf2
    S_7520_14 = 0.010177520 * nf2
    S_7520_22 = 0.025101925 * nf2
    S_4010 = 0.00425 * nf2
    # (effort, stiffness) per joint suffix/name
    def es(n):
        if n.endswith("hip_pitch_joint") or n.endswith("hip_roll_joint") or n.endswith("knee_joint"):
            return 139.0, S_7520_22
        if n.endswith("hip_yaw_joint"):
            return 88.0, S_7520_14
        if n.endswith("ankle_pitch_joint") or n.endswith("ankle_roll_joint"):
            return 50.0, 2 * S_5020
        if n in ("waist_roll_joint", "waist_pitch_joint"):
            return 50.0, 2 * S_5020
        if n == "waist_yaw_joint":
            return 88.0, S_7520_14
        if n.endswith("wrist_pitch_joint") or n.endswith("wrist_yaw_joint"):
            return 5.0, S_4010
        # shoulder_*, elbow, wrist_roll
        return 25.0, S_5020
    sc = np.zeros(29, dtype=np.float32)
    for i, n in enumerate(ISAACLAB_JOINT_NAMES):
        e, s = es(n)
        sc[i] = 0.25 * e / s
    return sc


def default_joint_pos_isaaclab():
    d = np.zeros(29, dtype=np.float32)
    for i, n in enumerate(ISAACLAB_JOINT_NAMES):
        if n.endswith("hip_pitch_joint"):
            d[i] = -0.312
        elif n.endswith("knee_joint"):
            d[i] = 0.669
        elif n.endswith("ankle_pitch_joint"):
            d[i] = -0.363
        elif n.endswith("elbow_joint"):
            d[i] = 0.6
        elif n == "left_shoulder_roll_joint":
            d[i] = 0.2
        elif n == "left_shoulder_pitch_joint":
            d[i] = 0.2
        elif n == "right_shoulder_roll_joint":
            d[i] = -0.2
        elif n == "right_shoulder_pitch_joint":
            d[i] = 0.2
    return d


# Body-joint indices in the 43-d dataset state/wbc (modality.json), in
# mujoco/grouped order (hands removed): legs(0:12)+waist(12:15)+l_arm(15:22)+r_arm(29:36)
BODY_MUJOCO_IDX = list(range(0, 22)) + list(range(29, 36))   # 29 indices


def quat_diff_angvel(q_t, q_tm1, dt):
    def norm(q):
        n = np.linalg.norm(q)
        return q / n if n > 1e-8 else np.array([1., 0., 0., 0.])
    q0, q1 = norm(q_tm1), norm(q_t)
    w0, x0, y0, z0 = q0
    c = np.array([w0, -x0, -y0, -z0])
    w1, x1, y1, z1 = c
    w2, x2, y2, z2 = q1
    qr = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])
    qr = norm(qr)
    w = np.clip(qr[0], -1, 1)
    ang = 2*np.arccos(w)
    s = np.sqrt(max(1 - w*w, 1e-12))
    axis = qr[1:]/s if s > 1e-6 else np.zeros(3)
    return (axis*ang/max(dt, 1e-6)).astype(np.float32)


def hist_idx(t, H, newest_first):
    idxs = [max(0, t-k) for k in range(H)]
    return idxs if newest_first else idxs[::-1]


def load_ep(root, ep):
    df = pd.read_parquet(os.path.join(root, "data", "chunk-000", f"episode_{ep:06d}.parquet"))
    def st(c): return np.stack([df[c].iloc[i] for i in range(len(df))]).astype(np.float32)
    return dict(wbc=st("action.wbc"), token=st("action.motion_token"),
                state=st("observation.state"), grav=st("observation.projected_gravity"),
                root=st("observation.root_orientation"), n=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/shared_disk/users/hengtao.li/locomanip/data/pick_place")
    ap.add_argument("--onnx", default="/mnt/pfs/users/hengtao.li/locomanip/GR00T-WholeBodyControl/"
                    "gear_sonic_deploy/policy/release/model_decoder.onnx")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--max_frames", type=int, default=400)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    dt = 1.0/args.fps
    ep = load_ep(args.data_root, args.episode)
    T = min(ep["n"], args.max_frames)
    iso2mj = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF)
    default = default_joint_pos_isaaclab()
    act_scale = action_scale_isaaclab()

    # body channels in MUJOCO order, then reorder to ISAACLAB order for the decoder
    body_pos_mj = ep["state"][:, BODY_MUJOCO_IDX]          # [T,29] mujoco
    body_act_mj = ep["wbc"][:, BODY_MUJOCO_IDX]            # [T,29] mujoco (GT action)
    gt_body_mj = body_act_mj                               # compare target (mujoco)

    mj2iso = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)
    INMAPS = {"iso2mj": iso2mj, "ident": np.arange(29), "mj2iso": mj2iso}
    grav = ep["grav"]                                      # [T,3]
    angv = np.zeros((ep["n"], 3), dtype=np.float32)
    for t in range(1, ep["n"]):
        angv[t] = quat_diff_angvel(ep["root"][t], ep["root"][t-1], dt)

    def assemble(t, H, newest_first, rel, use_angv, bp, bv, ba):
        h = hist_idx(t, H, newest_first)
        jp = bp[h]
        if rel:
            jp = jp - default[None]
        av = angv[h] if use_angv else np.zeros((H, 3), np.float32)
        parts = [ep["token"][t].reshape(-1), av.reshape(-1), jp.reshape(-1),
                 bv[h].reshape(-1), ba[h].reshape(-1), grav[h].reshape(-1)]
        return np.concatenate(parts).astype(np.float32)

    base_mae = float(np.mean(np.abs(gt_body_mj[:T] - body_pos_mj[:T])))
    print(f"episode {args.episode}  T={T}  baseline(wbc vs state) MAE={base_mae:.4f}\n")

    # Locked from prior sweep: inmap=mj2iso, rel=True, scaled output, H=10.
    # Now sweep the *derived* channels (ang_vel on/off, joint_vel scale, newest_first).
    perm = INMAPS["mj2iso"]
    bp = body_pos_mj[:, perm]
    ba = body_act_mj[:, perm]
    print(f"{'newest':>7} {'angv':>5} {'velscale':>8} | {'scaledMAE':>9}")
    best = None
    for nf, ua, vscale in itertools.product([True, False], [True, False], [1.0, 0.0, 0.05]):
        bv = np.zeros_like(bp); bv[1:] = (bp[1:] - bp[:-1]) / dt
        bv = bv * vscale
        dec = np.zeros((T, 29), np.float32)
        for t in range(T):
            obs = assemble(t, 10, nf, True, ua, bp, bv, ba)
            dec[t] = sess.run(None, {in_name: obs[None]})[0][0]
        target = default[None] + act_scale[None] * dec
        scaled_mae = float(np.mean(np.abs(target - ba[:T])))
        print(f"{str(nf):>7} {str(ua):>5} {vscale:>8} | {scaled_mae:>9.4f}")
        if best is None or scaled_mae < best[0]:
            best = (scaled_mae, "mj2iso", True, nf, "scaled", target.copy(), ba[:T].copy(), perm)

    if best:
        m, inmap_name, rel, nf, tag, pred, gt, perm = best
        print(f"\nBEST: MAE={m:.4f}  inmap={inmap_name} rel={rel} newest_first={nf} "
              f"interp={tag}  (baseline hold-pose={base_mae:.4f})")
        # per-joint correlation: high everywhere => scaling issue; mixed => ordering
        names_perm = [ISAACLAB_JOINT_NAMES[i] for i in perm]
        print("\nper-joint  corr   |pred-gt|.mean   name")
        corrs = []
        for j in range(29):
            a, b = pred[:, j], gt[:, j]
            c = np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
            corrs.append(c)
            print(f"  {j:2d}  {c:6.3f}   {np.mean(np.abs(a-b)):10.4f}     {names_perm[j]}")
        cc = np.array([c for c in corrs if not np.isnan(c)])
        print(f"\nmean|corr|={np.nanmean(np.abs(np.array(corrs))):.3f}  "
              f"frac corr>0.8: {np.mean(cc > 0.8):.2f}")


if __name__ == "__main__":
    main()
