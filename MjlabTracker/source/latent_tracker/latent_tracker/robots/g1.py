"""Unitree G1 robot config for latent tracking."""

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg

G1_CYLINDER_CFG = get_g1_robot_cfg

__all__ = ["G1_ACTION_SCALE", "G1_CYLINDER_CFG", "get_g1_robot_cfg"]
