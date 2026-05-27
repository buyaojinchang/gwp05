import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from world_action_model.trainer import EMA, ModuleDict, Trainer


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.video = nn.Linear(3, 3)
        self.action = nn.Linear(3, 3)
        for param in self.action.parameters():
            param.requires_grad = False


class TinyTrainer(Trainer):
    def get_models(self, model_config):
        raise NotImplementedError

    def forward_step(self, batch_dict):
        raise NotImplementedError


def test_fp32_ema_tracks_only_trainable_and_saves_full_checkpoint():
    model = ModuleDict({"transformer": TinyTransformer().to(dtype=torch.bfloat16)})
    ema = EMA(model, decay=0.5, device="model")

    assert ema.shadow
    assert all(t.dtype == torch.float32 for t in ema.shadow.values())
    assert all(name.startswith("transformer.video.") for name in ema.tracked_names)
    assert not any(name.startswith("transformer.action.") for name in ema.tracked_names)

    with torch.no_grad():
        for param in model["transformer"].video.parameters():
            param.add_(1.0)
    ema.update(model)
    assert ema.updates == 1

    trainer = TinyTrainer({"train": {"mixed_precision": "bf16"}})
    trainer.model = model
    trainer.cur_step = 3

    with tempfile.TemporaryDirectory() as tmp:
        trainer._save_checkpoint(tmp, ema)
        ckpt = Path(tmp) / "checkpoint-3"
        raw = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
        ema_model = torch.load(ckpt / "model_ema.pt", map_location="cpu", weights_only=False)
        ema_state = torch.load(ckpt / "ema_state.pt", map_location="cpu", weights_only=False)
        meta = json.loads((ckpt / "checkpoint_meta.json").read_text())

        assert set(raw) == set(model.state_dict())
        assert set(ema_model) == set(model.state_dict())
        assert "transformer.action.weight" in ema_model
        assert raw["transformer.video.weight"].dtype == torch.float32
        assert ema_model["transformer.video.weight"].dtype == torch.float32
        assert ema_model["transformer.action.weight"].dtype == torch.bfloat16
        assert ema_state["updates"] == 1
        assert all(t.dtype == torch.float32 for t in ema_state["shadow"].values())
        assert meta["ema_enabled"] is True
        assert meta["ema_fp32_trainable"] is True

        resumed_model = ModuleDict({"transformer": TinyTransformer().to(dtype=torch.bfloat16)})
        resumed_ema = EMA(resumed_model, decay=0.1, device="model")
        resumed_trainer = TinyTrainer({"train": {"mixed_precision": "bf16"}})
        resumed_trainer.model = resumed_model
        step = resumed_trainer._try_resume(tmp, resumed_ema)
        assert step == 3
        assert resumed_ema.decay == 0.5
        assert resumed_ema.updates == 1
        assert all(t.dtype == torch.float32 for t in resumed_ema.shadow.values())
