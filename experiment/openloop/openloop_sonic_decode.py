"""Open-loop eval: MoT sonic-latent prediction -> SONIC decoder -> joint action.

The locomanip ``pick_place g1_sonic`` MoT model predicts a 66-d sonic latent
(``motion_token[64]`` + ``hand_binary[2]``).  By itself the latent is *not* a
joint command -- in deployment the 64-d motion token is fed (together with the
robot proprioception) to the SONIC low-level whole-body-control policy (the
``UniversalTokenModule`` *decoder*, exported as ``model_decoder.onnx``), which
produces the body joint targets.  See:

  * ``GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml``
  * ``gear_sonic/utils/inference_helpers.py::export_universal_token_decoder_as_onnx``
  * ``gear_sonic/scripts/run_vla_inference.py`` (token published over ZMQ to the
    C++/TRT control loop that runs encoder+decoder).

This script closes that gap for an *open-loop* comparison against the dataset:

  1. predict the 66-d latent per window with the MoT model (same path as
     ``openloop_locomanip.py``);
  2. assemble the SONIC decoder proprioception **teacher-forced from the dataset**
     (joint pos/vel history, last-action history, base angular velocity, gravity
     direction) and feed the *predicted* motion token through ``model_decoder.onnx``;
  3. compare the decoded body joints against the ground-truth ``action.wbc`` and
     the predicted ``hand_binary`` against ground-truth ``action.hand_binary``.

Because the decoder needs proprioception (it is a closed-loop policy), the
comparison is "given the TRUE robot state, does the PREDICTED token decode to the
right joint action".  Proprioception is teacher-forced from the *raw* dataset
(the one that still carries ``action.wbc`` / ``observation.projected_gravity`` /
``observation.root_orientation`` -- not the repacked ``pick_place_gwp``).

IMPORTANT -- fidelity assumptions that must be validated against the real ONNX +
the gear_sonic obs definitions (flagged inline as ``ASSUMPTION``):
  * proprioception term order / dims (taken from observation_config.yaml);
  * history length and history stacking direction (newest-first vs oldest-first);
  * body-joint subset/order within the 43-d state and ``action.wbc``;
  * whether ``joint_pos`` is raw or relative-to-default (joint_pos_rel);
  * base-angular-velocity is *derived* from root_orientation finite differences
    (the raw dataset has no explicit angular-velocity channel).

Run ``--inspect_onnx`` first to print the decoder's real input/output spec.
"""

import argparse
import json
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/shared_disk/models/huggingface")

import numpy as np
import torch

# Reuse the MoT loader / sampler / preprocessing from the latent open-loop.
from openloop_locomanip import (  # noqa: E402
    build_model, sample_action, build_state, load_norm_stats,
    plot_action_comparison, EGO_VIEW_KEY,
)
from openloop_eval import preprocess_image, load_t5_embedding  # noqa: E402

logger = logging.getLogger(__name__)

# ── Raw-dataset column names (the dataset that still carries action.wbc) ──────
WBC_KEY = "action.wbc"                       # GT joint action [43]
MOTION_TOKEN_KEY = "action.motion_token"     # GT sonic latent token [64]
HAND_BINARY_KEY = "action.hand_binary"       # GT hand open/close [2]
JOINT_STATE_KEY = "observation.state"        # joint state [43]
GRAVITY_KEY = "observation.projected_gravity"  # gravity dir in base frame [3]
ROOT_ORI_KEY = "observation.root_orientation"  # base quaternion [4] (wxyz)

MOTION_TOKEN_DIM = 64
HAND_BINARY_DIM = 2

# ── G1 joint metadata (from gear_sonic/envs/manager_env/robots/g1.py) ─────────
# The decoder operates in isaaclab joint order; the dataset state/wbc are in the
# 43-d "grouped" order (modality.json). Validated against the GT token via
# calib_sonic_decode.py: dataset body channels remapped through MJ2ISO match the
# decoder best (MAE ~0.14 == hold-pose baseline).
MJ2ISO = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
          24, 18, 25, 19, 26, 20, 27, 21, 28]
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
# Body-joint indices in the 43-d dataset state/wbc (hands removed), grouped order:
# legs(0:12) + waist(12:15) + left_arm(15:22) + right_arm(29:36) = 29.
BODY_DATASET_IDX = list(range(0, 22)) + list(range(29, 36))
HAND_DATASET_IDX = list(range(22, 29)) + list(range(36, 43))  # 14 hand joints


def g1_default_joint_pos():
    """default_joint_pos in isaaclab order (init_state in robots/g1.py)."""
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


def g1_action_scale():
    """JointPositionAction scale = 0.25 * effort_limit / stiffness (isaaclab order)."""
    nf2 = (10 * 2 * np.pi) ** 2
    S_5020, S_7520_14, S_7520_22, S_4010 = (0.003609725 * nf2, 0.010177520 * nf2,
                                            0.025101925 * nf2, 0.00425 * nf2)

    def es(n):
        if n.endswith(("hip_pitch_joint", "hip_roll_joint", "knee_joint")):
            return 139.0, S_7520_22
        if n.endswith("hip_yaw_joint") or n == "waist_yaw_joint":
            return 88.0, S_7520_14
        if n.endswith(("ankle_pitch_joint", "ankle_roll_joint")) or n in ("waist_roll_joint", "waist_pitch_joint"):
            return 50.0, 2 * S_5020
        if n.endswith(("wrist_pitch_joint", "wrist_yaw_joint")):
            return 5.0, S_4010
        return 25.0, S_5020  # shoulder_*, elbow, wrist_roll

    sc = np.zeros(29, dtype=np.float32)
    for i, n in enumerate(ISAACLAB_JOINT_NAMES):
        e, s = es(n)
        sc[i] = 0.25 * e / s
    return sc

# ── SONIC decoder proprioception layout (verified against model_decoder.onnx) ─
# The released decoder ONNX input ``obs_dict`` is 994-d, output ``action`` is 29-d.
#   994 = token_state(64) + base_ang_vel(30) + joint_pos(290)
#       + joint_vel(290) + last_actions(290) + gravity_dir(30)
# i.e. history_length = 10 (matches sonic_release.yaml actor_prop_history_length=10;
# the deploy observation_config.yaml "436" comment is stale), body joints = 29
# (43-d action.wbc = 29 body + 14 hand). Per-frame dims: ang_vel/gravity=3, joints=29.
DECODER_PROPRIO_ORDER = [
    "token_state",
    "his_base_angular_velocity",
    "his_body_joint_positions",
    "his_body_joint_velocities",
    "his_last_actions",
    "his_gravity_dir",
]


def quat_to_ang_vel_base(quat_t, quat_tm1, dt):
    """Body-frame angular velocity from consecutive base quaternions (wxyz).

    ASSUMPTION: root_orientation is stored as wxyz in the world frame; we take a
    finite-difference relative rotation and express it in the body frame. This is
    an approximation for the deploy ``base_ang_vel`` term (the raw dataset has no
    explicit angular-velocity channel).
    """
    def normalize(q):
        n = np.linalg.norm(q)
        return q / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    q0 = normalize(np.asarray(quat_tm1, dtype=np.float64))
    q1 = normalize(np.asarray(quat_t, dtype=np.float64))
    # relative rotation q_rel = conj(q0) * q1  (rotation from t-1 to t in body frame)
    w0, x0, y0, z0 = q0
    conj0 = np.array([w0, -x0, -y0, -z0])
    w1, x1, y1, z1 = conj0
    w2, x2, y2, z2 = q1
    q_rel = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    q_rel = normalize(q_rel)
    w = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 1e-12))
    axis = q_rel[1:] / s if s > 1e-6 else np.zeros(3)
    return (axis * angle / max(dt, 1e-6)).astype(np.float32)


# ── SONIC decoder (ONNX) ─────────────────────────────────────────────────────

class SonicDecoder:
    """Thin onnxruntime wrapper around ``model_decoder.onnx``.

    The decoder maps a flat proprioception vector (token + history) to body joint
    targets. We treat the real ONNX as the source of truth for the input layout:
    ``input_dim`` is read from the graph and used to validate the assembled vector.
    """

    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        if providers is None:
            avail = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()
        self.in_name = self.inputs[0].name
        self.in_shape = list(self.inputs[0].shape)
        self.out_name = self.outputs[0].name
        # last static dim of the (possibly dynamic-batch) input
        self.input_dim = next((int(d) for d in reversed(self.in_shape)
                               if isinstance(d, int) and d > 0), None)

    def describe(self):
        lines = ["SONIC decoder ONNX I/O:"]
        for i in self.inputs:
            lines.append(f"  in : {i.name}  shape={i.shape}  dtype={i.type}")
        for o in self.outputs:
            lines.append(f"  out: {o.name}  shape={o.shape}  dtype={o.type}")
        return "\n".join(lines)

    def __call__(self, actor_obs):
        x = np.asarray(actor_obs, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        out = self.sess.run([self.out_name], {self.in_name: x})[0]
        return np.asarray(out, dtype=np.float32)


# ── Proprioception assembly (teacher-forced from the raw dataset) ─────────────

def _history_indices(t, hist_len, newest_first):
    """Frame indices [t-(H-1) .. t] clamped to >=0, ordered per convention."""
    idxs = [max(0, t - k) for k in range(hist_len)]   # newest..oldest
    return idxs if newest_first else list(reversed(idxs))


def assemble_proprio(t, token_pred, joint_pos, joint_vel, last_actions, ang_vel,
                     gravity, body_idx, hist_len, newest_first,
                     joint_pos_offset=None):
    """Build the flat decoder proprioception vector for frame ``t``.

    Args are per-frame arrays already restricted to the relevant signals:
      joint_pos/joint_vel/last_actions: [T, 29] (body joints)
      ang_vel/gravity:                  [T, 3]
    Order follows ``observation_config.yaml`` (release).
    """
    h = _history_indices(t, hist_len, newest_first)

    jp = joint_pos[h]                          # [H, 29]
    if joint_pos_offset is not None:           # ASSUMPTION: joint_pos_rel option
        jp = jp - joint_pos_offset[None]
    parts = {
        "token_state": np.asarray(token_pred, dtype=np.float32).reshape(-1),
        "his_base_angular_velocity": ang_vel[h].reshape(-1),
        "his_body_joint_positions": jp.reshape(-1),
        "his_body_joint_velocities": joint_vel[h].reshape(-1),
        "his_last_actions": last_actions[h].reshape(-1),
        "his_gravity_dir": gravity[h].reshape(-1),
    }
    return np.concatenate([parts[k] for k in DECODER_PROPRIO_ORDER]).astype(np.float32)


def load_raw_episode(data_root, episode_idx):
    """Load a raw pick_place episode that still carries action.wbc + proprio."""
    import pandas as pd
    parquet = os.path.join(data_root, "data", "chunk-000",
                           f"episode_{episode_idx:06d}.parquet")
    df = pd.read_parquet(parquet)

    def stack(col):
        if col not in df.columns:
            raise KeyError(f"Column {col!r} missing in {parquet}. "
                           f"Available: {list(df.columns)}")
        return np.stack([df[col].iloc[i] for i in range(len(df))]).astype(np.float32)

    wbc = stack(WBC_KEY)                 # [T, 43]  GT joint action
    motion = stack(MOTION_TOKEN_KEY)     # [T, 64]
    hand = stack(HAND_BINARY_KEY)        # [T, 2]
    state = stack(JOINT_STATE_KEY)       # [T, 43]
    gravity = stack(GRAVITY_KEY)         # [T, 3]
    root_ori = stack(ROOT_ORI_KEY) if ROOT_ORI_KEY in df.columns else None
    latent = np.concatenate([motion, hand], axis=1)  # [T, 66] sonic latent
    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0

    video_path = os.path.join(data_root, "videos", "chunk-000", EGO_VIEW_KEY,
                              f"episode_{episode_idx:06d}.mp4")
    frames = None
    if os.path.exists(video_path):
        import imageio
        reader = imageio.get_reader(video_path)
        frames = [np.array(f) for f in reader]
        reader.close()

    return dict(wbc=wbc, motion=motion, hand=hand, state=state, gravity=gravity,
                root_ori=root_ori, latent=latent, frames=frames,
                num_frames=len(df), task_index=task_index)


def discover_episodes(data_root):
    data_dir = os.path.join(data_root, "data", "chunk-000")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    eps = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("episode_") and fn.endswith(".parquet"):
            eps.append(int(fn[len("episode_"):-len(".parquet")]))
    return eps


# ── Token prediction (per-window, dense via overlap averaging) ───────────────

@torch.no_grad()
def predict_latents(args, md, stats, ep, prompt_embeds):
    """Predict the dense 66-d sonic latent (token+hand) for every frame."""
    device, dtype = md["device"], md["dtype"]
    vae = md["vae"]
    latents_mean, latents_std = md["latents_mean"], md["latents_std"]
    T = ep["num_frames"]
    num_frames = args.num_frames
    state_token_dims = [int(args.joint_state_dim), int(args.latent_state_dim)]
    dst_w, dst_h = tuple(args.dst_size)
    n_video = len(ep["frames"])

    pred_sum = np.zeros((T, args.action_dim), dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    for start in range(0, T, args.replan_steps):
        frame_idx = min(start, n_video - 1)
        ref_image = preprocess_image(ep["frames"][frame_idx], (dst_w, dst_h)).to(device, dtype=dtype)
        ref_latents = vae.encode(ref_image.unsqueeze(2)).latent_dist.mode()
        ref_latents = (ref_latents - latents_mean) * latents_std

        state_np = build_state(ep["state"][start], ep["latent"][start], stats,
                               state_token_dims, args.state_dim)
        state = torch.from_numpy(state_np).to(device, dtype=dtype).unsqueeze(0)
        state_mask = torch.ones((1, state_np.shape[0]), dtype=torch.bool, device=device)

        acc = None
        for s_i in range(args.num_samples):
            if args.num_samples > 1:
                torch.manual_seed(args.seed + 1000 * s_i + start)
            one = sample_action(md, ref_latents, prompt_embeds, state, state_mask,
                                num_steps=args.num_steps, num_frames=num_frames)
            one = one.float().squeeze(0)
            acc = one if acc is None else acc + one
        pred = (acc / args.num_samples).cpu().numpy()

        end = min(start + num_frames, T)
        pred_sum[start:end] += pred[:end - start]
        counts[start:end] += 1

    mask = counts > 0
    pred_sum[mask] /= counts[mask, None]
    return pred_sum, mask


# ── Main eval loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run(args, md, stats, decoder):
    device, dtype = md["device"], md["dtype"]
    fallback_t5 = torch.zeros(1, 64, 4096, dtype=dtype, device=device)
    dt = 1.0 / args.fps
    # Validated decoder mapping (see calib_sonic_decode.py): take the 29 body
    # joints out of the 43-d state/wbc, then remap dataset -> decoder (isaaclab)
    # order via MJ2ISO. joint_pos is relative to the g1 default pose; the decoder
    # outputs a raw action turned into a joint target by default + action_scale*raw.
    body_idx = np.asarray(BODY_DATASET_IDX, dtype=np.int64)
    perm = np.asarray(MJ2ISO, dtype=np.int64)
    default_pos = g1_default_joint_pos()        # [29] isaaclab order
    act_scale = g1_action_scale()               # [29] isaaclab order
    n_body = len(body_idx)
    joint_names = [ISAACLAB_JOINT_NAMES[i] for i in perm]

    all_eps = discover_episodes(args.data_root)
    if args.episode_indices:
        episodes = [e for e in args.episode_indices if e in all_eps]
    else:
        episodes = all_eps[: args.max_episodes] if args.max_episodes > 0 else all_eps
    print(f"Found {len(all_eps)} raw episodes; evaluating {len(episodes)}: {episodes}")

    results = []
    for episode_idx in episodes:
        task_name = f"ep{episode_idx:06d}"
        print(f"\n{'='*60}\nEpisode: {episode_idx}")
        ep = load_raw_episode(args.data_root, episode_idx)
        if ep["frames"] is None:
            raise RuntimeError(f"Missing ego_view video for episode {episode_idx}")
        T = ep["num_frames"]
        print(f"  Frames: {T}  state={ep['state'].shape}  wbc={ep['wbc'].shape}")

        # The raw pick_place dir has no meta/t5_text_embeds.pt; load it from
        # --t5_root (the gwp dataset) keyed by task_index. Zeros fallback badly
        # degrades the token prediction (language conditioning was trained on).
        prompt_embeds = load_t5_embedding(args.t5_root, episode_idx, device, dtype,
                                          task_index=ep.get("task_index", 0))
        if prompt_embeds is None:
            print(f"  [WARN] No T5 embedding under {args.t5_root}; using zeros fallback "
                  "(this degrades the predicted token!)")
            prompt_embeds = fallback_t5
        else:
            print(f"  Loaded T5 embedding {tuple(prompt_embeds.shape)} (task {ep.get('task_index', 0)})")

        # 1) predict the dense sonic latent (token[64] + hand[2]) for all frames
        pred_latent, mask = predict_latents(args, md, stats, ep, prompt_embeds)
        pred_token = pred_latent[:, :MOTION_TOKEN_DIM]            # [T, 64]
        pred_hand = pred_latent[:, MOTION_TOKEN_DIM:args.action_dim]  # [T, 2]

        # token-space diagnostic: how well does the WM predict the latent itself?
        gt_token = ep["motion"]
        mtok = mask
        tok_mae = float(np.mean(np.abs(pred_token[mtok] - gt_token[mtok])))
        tok_corr = float(np.corrcoef(pred_token[mtok].ravel(), gt_token[mtok].ravel())[0, 1])
        print(f"  Token-space: pred-vs-GT MAE={tok_mae:.4f}  corr={tok_corr:.3f}  "
              f"| range pred[{pred_token[mtok].min():.2f},{pred_token[mtok].max():.2f}] "
              f"gt[{gt_token[mtok].min():.2f},{gt_token[mtok].max():.2f}]")

        # 2) teacher-forced proprioception channels (body joints, in decoder order)
        body_pos = ep["state"][:, body_idx][:, perm]             # [T, 29] decoder order
        body_act = ep["wbc"][:, body_idx][:, perm]               # GT body action *target* (rad)
        # The decoder's `last_actions` channel is the RAW policy action (network
        # output, pre scale/offset), NOT the joint target. isaaclab applies
        #   target = default + action_scale * raw  =>  raw = (target - default) / scale.
        # Feeding the target directly injects a constant default/scale offset into
        # this input channel, which the decoder echoes as a constant output bias.
        body_act_raw = (body_act - default_pos[None]) / act_scale[None]   # [T, 29] raw action
        body_vel = np.zeros_like(body_pos)
        body_vel[1:] = (body_pos[1:] - body_pos[:-1]) / dt        # finite-diff velocity
        gravity = ep["gravity"]                                  # [T, 3]
        ang_vel = np.zeros((T, 3), dtype=np.float32)
        if ep["root_ori"] is not None and not args.zero_ang_vel:
            for t in range(1, T):
                ang_vel[t] = quat_to_ang_vel_base(ep["root_ori"][t], ep["root_ori"][t - 1], dt)

        def decode_with_token(token_seq):
            """Decode every frame: token + teacher-forced proprio -> joint target."""
            out = np.zeros((T, n_body), dtype=np.float32)
            for t in range(T):
                obs = assemble_proprio(
                    t, token_seq[t], body_pos, body_vel, body_act_raw, ang_vel, gravity,
                    body_idx, args.history_length, args.history_newest_first,
                    joint_pos_offset=default_pos)
                if t == 0 and episode_idx == episodes[0] and token_seq is pred_token:
                    if decoder.input_dim is not None and obs.shape[0] != decoder.input_dim:
                        print(f"  [WARN] proprio dim {obs.shape[0]} != decoder input "
                              f"{decoder.input_dim}. Check history_length / body joints.")
                    else:
                        print(f"  proprio dim {obs.shape[0]} matches decoder input {decoder.input_dim}")
                raw = decoder(obs)[0, :n_body]                   # raw policy action (isaaclab)
                out[t] = default_pos + act_scale * raw            # -> joint target
            return out

        # 3) decode with the PREDICTED token, and (reference) with the GT token to
        #    show the decoder reconstruction floor independent of token prediction.
        decoded = decode_with_token(pred_token)
        decoded_gt = decode_with_token(ep["motion"]) if not args.skip_gt_ref else None

        # 4) metrics: decoded joint target vs GT action.wbc[body]; hand vs GT hand
        gt_body = body_act
        m = mask
        def mae(a, b): return float(np.mean(np.abs(a[m] - b[m])))
        body_mae = mae(decoded, gt_body)
        body_mse = float(np.mean((decoded[m] - gt_body[m]) ** 2))
        gt_token_mae = mae(decoded_gt, gt_body) if decoded_gt is not None else None
        # trivial baselines on the same frames
        zeros_mae = float(np.mean(np.abs(gt_body[m])))
        hold0_mae = float(np.mean(np.abs(gt_body[m] - gt_body[m][0:1])))
        state_mae = mae(gt_body, body_pos)                       # action vs current pose
        hand_mae = mae(pred_hand, ep["hand"])

        print(f"  Decoded(pred token) body MAE={body_mae:.6f}  MSE={body_mse:.6f}")
        if gt_token_mae is not None:
            print(f"  Decoded(GT token)  body MAE={gt_token_mae:.6f}  <- reconstruction floor")
        print(f"  Baselines MAE -> zeros: {zeros_mae:.4f}, hold-first: {hold0_mae:.4f}, "
              f"wbc-vs-pose: {state_mae:.4f}")
        print(f"  Hand binary MAE (pred vs gt): {hand_mae:.6f}")

        save_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_action_comparison(
            os.path.join(save_dir, f"{task_name}_joints.png"), task_name,
            gt_body, decoded, list(range(min(n_body, 16))), body_mse, body_mae,
            title_suffix=" body joints (decoded vs wbc)", names=joint_names)
        np.savez_compressed(os.path.join(save_dir, f"{task_name}_arrays.npz"),
                            decoded=decoded, decoded_gt_token=decoded_gt if decoded_gt is not None else np.empty(0),
                            gt_wbc=gt_body, pred_hand=pred_hand, gt_hand=ep["hand"],
                            joint_names=np.asarray(joint_names, dtype="U32"), mask=mask)
        metrics = dict(episode=episode_idx, num_frames=int(T), n_body=int(n_body),
                       body_mae=body_mae, body_mse=body_mse, gt_token_mae=gt_token_mae,
                       hand_mae=hand_mae, baseline_zeros_mae=zeros_mae,
                       baseline_hold_first_mae=hold0_mae, wbc_vs_pose_mae=state_mae)
        with open(os.path.join(save_dir, f"{task_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved to {save_dir}")
        results.append(metrics)

    if results:
        print(f"\n{'='*60}\n  Sonic-decode Open-Loop Summary ({len(results)} eps)\n{'='*60}")
        for r in results:
            gtt = f"{r['gt_token_mae']:.4f}" if r["gt_token_mae"] is not None else "n/a"
            print(f"  ep{r['episode']:06d}  bodyMAE={r['body_mae']:.6f}  "
                  f"GT-token floor={gtt}  handMAE={r['hand_mae']:.6f}  "
                  f"(hold-first={r['baseline_hold_first_mae']:.4f})")
        mean_body = float(np.mean([r["body_mae"] for r in results]))
        mean_hand = float(np.mean([r["hand_mae"] for r in results]))
        print(f"  mean bodyMAE={mean_body:.6f}  meanHandMAE={mean_hand:.6f}")
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(dict(checkpoint=args.checkpoint_path, data_root=args.data_root,
                           decoder_onnx=args.decoder_onnx, per_episode=results,
                           mean_body_mae=mean_body, mean_hand_mae=mean_hand), f, indent=2)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sonic-decode open-loop eval (latent -> joints vs wbc)")
    p.add_argument("--model_id", type=str,
                   default=os.environ.get("WAN22_DIFFUSERS_PATH",
                                          "/shared_disk/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"))
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--decoder_onnx", type=str,
                   default="/mnt/pfs/users/hengtao.li/locomanip/GR00T-WholeBodyControl/"
                           "gear_sonic_deploy/policy/release/model_decoder.onnx",
                   help="SONIC UniversalTokenModule decoder exported to ONNX")
    p.add_argument("--data_root", type=str,
                   default="/shared_disk/users/hengtao.li/locomanip/data/pick_place",
                   help="RAW lerobot dataset that still carries action.wbc + proprio "
                        "(NOT the repacked pick_place_gwp). Upload it here.")
    p.add_argument("--stats_path", type=str, default=None,
                   help="norm_stats for the MoT state tokens (defaults to "
                        "pick_place_gwp/norm_stats_delta.json)")
    p.add_argument("--t5_root", type=str,
                   default="/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp",
                   help="Dataset dir holding meta/t5_text_embeds.pt (raw pick_place lacks it). "
                        "T5 language embeddings are keyed by task_index.")
    p.add_argument("--output_dir", type=str, default=None)

    # MoT latent-prediction config (mirrors openloop_locomanip.py)
    p.add_argument("--action_dim", type=int, default=66)
    p.add_argument("--state_dim", type=int, default=66)
    p.add_argument("--joint_state_dim", type=int, default=43)
    p.add_argument("--latent_state_dim", type=int, default=66)
    p.add_argument("--num_frames", type=int, default=56)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_flow_shift", type=float, default=5.0)
    p.add_argument("--replan_steps", type=int, default=56)
    p.add_argument("--dst_size", type=int, nargs=2, default=[320, 256])
    p.add_argument("--action_expert_hidden_dim", type=int, default=1024)
    p.add_argument("--action_expert_ffn_dim", type=int, default=4096)
    p.add_argument("--mot_checkpoint_mixed_attn", action=argparse.BooleanOptionalAction, default=True)

    # SONIC decoder proprioception config (calibrated via calib_sonic_decode.py)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--history_length", type=int, default=10,
                   help="History length. Verified: 10 => 994-d decoder input.")
    p.add_argument("--history_newest_first", action=argparse.BooleanOptionalAction, default=False,
                   help="History stacking direction (calibration favored oldest-first).")
    p.add_argument("--zero_ang_vel", action="store_true", default=False,
                   help="Feed zero base angular velocity instead of the quaternion-derived one "
                        "(calibration showed it has negligible effect).")
    p.add_argument("--skip_gt_ref", action="store_true", default=False,
                   help="Skip the GT-token reconstruction-floor decode (faster).")

    p.add_argument("--episode_indices", type=int, nargs="*", default=None)
    p.add_argument("--max_episodes", type=int, default=3)
    p.add_argument("--inspect_onnx", action="store_true", default=False,
                   help="Just load the decoder ONNX, print its I/O spec, and exit.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    if not os.path.exists(args.decoder_onnx):
        raise FileNotFoundError(
            f"Decoder ONNX not found: {args.decoder_onnx}\n"
            "Download it from HF (nvidia/GEAR-SONIC) e.g.:\n"
            "  cd GR00T-WholeBodyControl && python download_from_hf.py\n"
            "or HF_ENDPOINT=https://hf-mirror.com python download_from_hf.py")

    decoder = SonicDecoder(args.decoder_onnx)
    print(decoder.describe())
    print(f"  inferred input_dim = {decoder.input_dim}")
    if args.inspect_onnx:
        return

    if args.stats_path is None:
        args.stats_path = "/shared_disk/users/hengtao.li/locomanip/data/pick_place_gwp/norm_stats_delta.json"
    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        model_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
        args.output_dir = os.path.join(
            "/shared_disk/users/hengtao.li/locomanip/exp/openloop_sonic_decode",
            os.path.basename(os.path.dirname(ckpt_dir)) or "exp", model_name)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Sonic-decode Open-Loop [latent -> decoder -> joint vs action.wbc]")
    print("=" * 60)
    print(f"  Checkpoint:   {args.checkpoint_path}")
    print(f"  Decoder ONNX: {args.decoder_onnx}")
    print(f"  Raw data:     {args.data_root}")
    print(f"  History len:  {args.history_length} (newest_first={args.history_newest_first})")
    print(f"  Body joints:  {len(BODY_DATASET_IDX)} (dataset->decoder remap via MJ2ISO)")
    print("=" * 60)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    md = build_model(
        args.model_id, args.checkpoint_path,
        action_dim=args.action_dim, state_dim=args.state_dim,
        state_token_dims=(args.joint_state_dim, args.latent_state_dim),
        device="cuda", dtype=torch.bfloat16,
        action_expert_hidden_dim=args.action_expert_hidden_dim,
        action_expert_ffn_dim=args.action_expert_ffn_dim,
        mot_checkpoint_mixed_attn=args.mot_checkpoint_mixed_attn)
    md["action_flow_shift"] = args.action_flow_shift
    stats = load_norm_stats(args.stats_path)

    run(args, md, stats, decoder)


if __name__ == "__main__":
    main()
