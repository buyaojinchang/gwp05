import torch

from world_action_model.models.action_state_dit import ActionStateDiT
from world_action_model.models.transformer_wa_casual import CasualWorldActionTransformer
from world_action_model.models.transformer_wa_mot import MoTWorldActionTransformer


def build_tiny_model():
    video = CasualWorldActionTransformer(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=32,
        num_layers=2,
        rope_max_seq_len=16,
    )
    action = ActionStateDiT(
        action_dim=3,
        state_dim=5,
        hidden_dim=8,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        rope_max_seq_len=16,
    )
    return MoTWorldActionTransformer(video, action, mot_checkpoint_mixed_attn=False)


def test_video_expert_legacy_action_modules_are_dropped():
    model = build_tiny_model()
    legacy_attrs = (
        "action_rope",
        "action_encoder",
        "action_decoder",
        "state_encoder",
        "condition_embedder_action",
    )
    for attr in legacy_attrs:
        assert not hasattr(model.video_expert, attr)
    forbidden_prefixes = tuple(f"mot.mixtures.video.{attr}." for attr in legacy_attrs)
    assert not any(name.startswith(forbidden_prefixes) for name in model.state_dict())


def test_gwp_casual_mask_semantics():
    mask = MoTWorldActionTransformer.build_gwp_casual_mask(
        num_state_tokens=1,
        num_ref_tokens=4,
        num_action_tokens=3,
        num_noisy_tokens=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state_ref_end = 5
    action_end = 8
    assert torch.isneginf(mask[:state_ref_end, state_ref_end:]).all()
    assert torch.isneginf(mask[state_ref_end:action_end, action_end:]).all()
    assert (mask[action_end:, :] == 0).all()
    assert (mask[state_ref_end:action_end, :action_end] == 0).all()


def test_tiny_forward_full_and_action_only():
    torch.manual_seed(0)
    model = build_tiny_model()
    batch = 2
    ref = torch.randn(batch, 4, 1, 4, 4)
    noisy = torch.randn(batch, 4, 1, 4, 4)
    state = torch.randn(batch, 1, 5)
    action = torch.randn(batch, 3, 3)
    prompt = torch.randn(batch, 4, 16)

    # token order: [state | ref_video | action | noisy_video]
    timestep = torch.zeros(batch, 1 + 4 + 3 + 4)
    timestep[:, 1 + 4 :] = 500
    video_pred, action_pred = model(
        ref_latents=ref,
        noisy_latents=noisy,
        timestep=timestep,
        encoder_hidden_states=prompt,
        state=state,
        action=action,
        return_dict=False,
    )
    assert tuple(video_pred.shape) == (batch, 4, 2, 4, 4)
    assert tuple(action_pred.shape) == (batch, 3, 3)
    assert torch.isfinite(video_pred).all()
    assert torch.isfinite(action_pred).all()

    action_only = model(
        ref_latents=ref,
        noisy_latents=noisy,
        timestep=timestep,
        encoder_hidden_states=prompt,
        state=state,
        action=action,
        action_only=True,
        return_dict=False,
    )
    assert tuple(action_only.shape) == (batch, 3, 3)
    assert torch.isfinite(action_only).all()
