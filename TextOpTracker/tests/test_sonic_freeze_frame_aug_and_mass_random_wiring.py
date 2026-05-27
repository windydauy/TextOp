from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sonic_freeze_frame_aug_and_mass_randomization_are_wired() -> None:
    commands_src = (
        REPO_ROOT
        / "TextOpTracker"
        / "source"
        / "textop_tracker"
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "mdp"
        / "commands_multi.py"
    ).read_text(encoding="utf-8")
    tracking_cfg_src = (
        REPO_ROOT
        / "TextOpTracker"
        / "source"
        / "textop_tracker"
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "tracking_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert "freeze_frame_aug: bool = False" in commands_src
    assert "freeze_frame_aug_prob: float = 0.1" in commands_src
    assert "freeze_frame_aug=True" in tracking_cfg_src

    update_buffers_body = commands_src.split("def _update_buffers", 1)[1].split(
        "\n    @property\n    def command", 1
    )[0]
    assert update_buffers_body.index("if self.cfg.freeze_frame_aug") < update_buffers_body.index(
        "if self.cfg.random_static_prob > 0"
    )
    assert "freeze_mask = torch.rand(" in update_buffers_body
    assert "< self.cfg.freeze_frame_aug_prob" in update_buffers_body
    assert "self.metrics[\"freeze_frame_aug_ratio\"][env_ids] = 0.0" in update_buffers_body
    assert "self.metrics[\"freeze_frame_aug_ratio\"][freeze_env_ids] = 1.0" in update_buffers_body
    assert "self.joint_pos_buffer[env_id, idx:] = self.joint_pos_buffer[" in update_buffers_body
    assert "self.joint_vel_buffer[env_id, idx:] = 0.0" in update_buffers_body
    assert "self.body_pos_w_buffer[env_id, idx:] = self.body_pos_w_buffer[" in update_buffers_body
    assert "self.body_quat_w_buffer[env_id, idx:] = self.body_quat_w_buffer[" in update_buffers_body
    assert "self.body_lin_vel_w_buffer[env_id, idx:] = 0.0" in update_buffers_body
    assert "self.body_ang_vel_w_buffer[env_id, idx:] = 0.0" in update_buffers_body
    assert "self.motion_ae_latent_buffer[env_id, idx:]" in update_buffers_body
    assert "self.motion_transformer_vae_latent_buffer[env_id, idx:]" in update_buffers_body

    event_cfg_body = tracking_cfg_src.split("class EventCfg", 1)[1].split("\n\n@configclass", 1)[0]
    assert "randomize_rigid_body_mass = EventTerm(" in event_cfg_body
    assert "func=mdp.randomize_rigid_body_mass" in event_cfg_body
    assert 'mode="startup"' in event_cfg_body
    assert 'SceneEntityCfg("robot", body_names=".*wrist_yaw.*|torso_link")' in event_cfg_body
    assert '"mass_distribution_params": (0.8, 2.5)' in event_cfg_body
    assert '"operation": "scale"' in event_cfg_body
