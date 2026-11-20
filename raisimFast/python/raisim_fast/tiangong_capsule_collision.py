"""Sim2Real-calibrated Tiangong3 analytic capsule/sphere collision profile for Raisim2_FAST.

Provides exact geometry parity with ``MLAB4_Fast``'s ``tiangong3_capsule_v1`` profile so
dual-simulator GPU+CPU training (``MLAB4_Fast`` + ``Raisim2_FAST`` / ``2SIM_TRAIN``) and
standalone RaiSim CPU rollouts use identical contact envelopes, 1.35-deg heel-to-toe foot
sole geometry, calibrated hand TCP capsules, and 2-hop compound-joint self-collision
exclusions on ``tiangong3_v2_29_head_fixed``.
"""
from __future__ import annotations

import math
from typing import Literal
import xml.etree.ElementTree as ET

import numpy as np

TIANGONG3_CAPSULE_PROFILE_VERSION = "tiangong3_capsule_v1"

DEFAULT_FRICTION: tuple[float, float, float] = (1.0, 0.01, 0.01)

TIANGONG3_CAPSULE_FOOT_GEOM_NAMES: tuple[str, ...] = (
    "foot_l_1_collision",
    "foot_l_2_collision",
    "foot_l_3_collision",
    "foot_l_4_collision",
    "foot_r_1_collision",
    "foot_r_2_collision",
    "foot_r_3_collision",
    "foot_r_4_collision",
)

# 2-hop compound-joint self-collision exclusions (aligned with MLAB4_Fast tiangong3_capsule_v1)
TIANGONG3_CAPSULE_EXTRA_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("pelvis", "hip_roll_l_link"),
    ("pelvis", "hip_roll_r_link"),
    ("pelvis", "hip_yaw_l_link"),
    ("pelvis", "hip_yaw_r_link"),
    ("pelvis", "waist_pitch_link"),
    ("hip_pitch_l_link", "hip_yaw_l_link"),
    ("hip_pitch_r_link", "hip_yaw_r_link"),
    ("waist_pitch_link", "shoulder_roll_l_link"),
    ("waist_pitch_link", "shoulder_roll_r_link"),
    ("shoulder_roll_l_link", "elbow_pitch_l_link"),
    ("shoulder_roll_r_link", "elbow_pitch_r_link"),
    ("elbow_pitch_l_link", "wrist_pitch_l_link"),
    ("elbow_pitch_r_link", "wrist_pitch_r_link"),
    ("elbow_yaw_l_link", "left_tcp_link"),
    ("elbow_yaw_r_link", "right_tcp_link"),
    ("wrist_pitch_l_link", "left_tcp_link"),
    ("wrist_pitch_r_link", "right_tcp_link"),
)

# Internal / sensor / intermediate gimbal bodies stripped of redundant collision primitives
TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES: tuple[str, ...] = (
    "imu_waist_link",
    "imu_head_link",
    "camera_body_front_link",
    "camera_head_link",
    "radar_head_link",
    "head_pitch_link",
    "waist_yaw_link",
    "waist_roll_link",
    "ankle_pitch_l_link",
    "ankle_pitch_r_link",
    "shoulder_pitch_l_link",
    "shoulder_pitch_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
)

PrimitiveSpec = tuple[
    Literal["capsule", "sphere"],
    str,  # geom_name
    float,  # radius
    tuple[float, float, float],  # p1 (capsule) or pos (sphere)
    tuple[float, float, float] | None,  # p2 (capsule) or None (sphere)
    int,  # condim
]


def _build_body_primitives() -> dict[str, tuple[PrimitiveSpec, ...]]:
    specs: dict[str, list[PrimitiveSpec]] = {
        "pelvis": [
            ("capsule", "pelvis_l_collision", 0.080, (-0.028, 0.045, -0.036), (0.026, 0.045, -0.036), 1),
            ("capsule", "pelvis_r_collision", 0.080, (-0.028, -0.045, -0.036), (0.026, -0.045, -0.036), 1),
            ("capsule", "pelvis_lower_collision", 0.074, (-0.002, -0.050, -0.108), (-0.002, 0.050, -0.108), 1),
        ],
        "waist_pitch_link": [
            ("capsule", "torso_lower_collision", 0.108, (0.002, -0.055, 0.105), (0.002, 0.055, 0.105), 1),
            ("capsule", "torso_upper_collision", 0.106, (-0.001, -0.055, 0.220), (-0.001, 0.055, 0.220), 1),
            ("sphere", "head_crown_collision", 0.064, (0.0002, 0.0, 0.464), None, 1),
        ],
        "head_yaw_link": [
            ("sphere", "head_yaw_link_collision", 0.085, (0.001, 0.0, 0.008), None, 1),
        ],
    }

    for side, sy in (("l", 1.0), ("r", -1.0)):
        specs[f"hip_pitch_{side}_link"] = [
            (
                "capsule",
                f"hip_pitch_{side}_link_collision",
                0.049,
                (-0.001, 0.052 * sy, -0.042),
                (0.002, 0.090 * sy, -0.024),
                1,
            ),
        ]
        specs[f"hip_roll_{side}_link"] = [
            (
                "capsule",
                f"hip_roll_{side}_link_collision",
                0.058,
                (0.001, 0.0, 0.002),
                (0.001, 0.0, -0.030),
                1,
            ),
        ]
        specs[f"hip_yaw_{side}_link"] = [
            (
                "capsule",
                f"hip_yaw_{side}_link_collision",
                0.066,
                (0.001, -0.006 * sy, -0.060),
                (0.001, -0.006 * sy, -0.205),
                1,
            ),
            (
                "capsule",
                f"thigh_distal_{side}_collision",
                0.051,
                (-0.002, 0.0, -0.220),
                (-0.006, 0.0, -0.292),
                1,
            ),
        ]
        specs[f"knee_pitch_{side}_link"] = [
            (
                "capsule",
                f"knee_cap_{side}_collision",
                0.054,
                (0.017, 0.0, -0.015),
                (0.016, 0.0, -0.175),
                1,
            ),
            (
                "capsule",
                f"shin_lower_{side}_collision",
                0.038,
                (0.011, 0.0, -0.195),
                (0.007, 0.0, -0.355),
                1,
            ),
        ]
        # 4 tapered foot sole capsules matching exact CAD 1.35-deg heel-to-toe slope
        specs[f"ankle_roll_{side}_link"] = [
            ("capsule", f"foot_{side}_1_collision", 0.010, (-0.084, -0.0335, -0.04677), (0.085, -0.0455, -0.04305), 3),
            ("capsule", f"foot_{side}_2_collision", 0.010, (-0.084, 0.0335, -0.04677), (0.085, 0.0455, -0.04305), 3),
            ("capsule", f"foot_{side}_3_collision", 0.010, (0.085, -0.0455, -0.04305), (0.1688, -0.0180, -0.04050), 3),
            ("capsule", f"foot_{side}_4_collision", 0.010, (0.085, 0.0455, -0.04305), (0.1688, 0.0180, -0.04050), 3),
        ]
        specs[f"shoulder_roll_{side}_link"] = [
            (
                "capsule",
                f"shoulder_roll_{side}_link_collision",
                0.048,
                (-0.001, 0.0, -0.009),
                (-0.001, 0.0, -0.056),
                1,
            ),
        ]
        specs[f"shoulder_yaw_{side}_link"] = [
            (
                "capsule",
                f"shoulder_yaw_{side}_link_collision",
                0.041,
                (0.001, 0.0, -0.020),
                (0.012, -0.002 * sy, -0.154),
                1,
            ),
        ]
        specs[f"elbow_pitch_{side}_link"] = [
            (
                "capsule",
                f"elbow_pitch_{side}_link_collision",
                0.045,
                (-0.009, -0.001 * sy, 0.0),
                (-0.018, 0.0, -0.075),
                1,
            ),
        ]
        specs[f"elbow_yaw_{side}_link"] = [
            (
                "capsule",
                f"elbow_yaw_{side}_link_collision",
                0.036,
                (0.0, 0.003 * sy, -0.015),
                (0.0, 0.0, -0.092),
                1,
            ),
        ]
        specs[f"wrist_pitch_{side}_link"] = [
            (
                "capsule",
                f"wrist_pitch_{side}_link_collision",
                0.034,
                (0.0, -0.002 * sy, 0.0),
                (-0.001, 0.0, -0.060),
                1,
            ),
        ]
        tcp_body = "left_tcp_link" if side == "l" else "right_tcp_link"
        specs[tcp_body] = [
            (
                "capsule",
                f"{tcp_body}_collision",
                0.047,
                (0.005, -0.006 * sy, -0.018),
                (0.006, -0.008 * sy, -0.0576),
                3,
            ),
        ]

    return {k: tuple(v) for k, v in specs.items()}


TIANGONG3_CAPSULE_BODY_PRIMITIVES: dict[str, tuple[PrimitiveSpec, ...]] = _build_body_primitives()

_TIANGONG3_SIGNATURE_BODIES: frozenset[str] = frozenset(
    {
        "pelvis",
        "waist_pitch_link",
        "ankle_roll_l_link",
        "ankle_roll_r_link",
        "left_tcp_link",
        "right_tcp_link",
    }
)


def is_tiangong3_body_set(body_names: set[str] | frozenset[str]) -> bool:
    """Return True when the articulated body set matches ``tiangong3_v2_29_head_fixed``."""
    return _TIANGONG3_SIGNATURE_BODIES.issubset(body_names)


def _mat_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    if sy > 1e-7:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0
    return (roll, pitch, yaw)


def capsule_fromto_to_urdf_origin(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Convert MuJoCo ``fromto`` capsule endpoints to URDF ``(xyz, rpy, length)`` (Z-axis aligned)."""
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    center = 0.5 * (a + b)
    diff = b - a
    length = float(np.linalg.norm(diff))
    if length < 1e-12:
        return (float(center[0]), float(center[1]), float(center[2])), (0.0, 0.0, 0.0), 1e-4
    z = diff / length
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return (float(center[0]), float(center[1]), float(center[2])), _mat_to_rpy(R), length


def emit_tiangong3_capsule_urdf_collisions(link_el: ET.Element, body_name: str) -> int:
    """Append Sim2Real-calibrated Tiangong3 capsule/sphere ``<collision>`` elements to a URDF ``<link>``.

    Returns the number of analytical primitives emitted (0 for stripped internal/sensor links).
    """
    if body_name in TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES:
        return 0
    prims = TIANGONG3_CAPSULE_BODY_PRIMITIVES.get(body_name)
    if not prims:
        return 0

    for kind, geom_name, radius, p1, p2, _condim in prims:
        col = ET.SubElement(link_el, "collision", name=geom_name)
        if kind == "sphere":
            ET.SubElement(
                col,
                "origin",
                xyz=f"{p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}",
                rpy="0.000000 0.000000 0.000000",
            )
            geom = ET.SubElement(col, "geometry")
            ET.SubElement(geom, "sphere", radius=f"{radius:.6f}")
        else:
            assert p2 is not None
            xyz, rpy, length = capsule_fromto_to_urdf_origin(p1, p2)
            ET.SubElement(
                col,
                "origin",
                xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
                rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}",
            )
            geom = ET.SubElement(col, "geometry")
            ET.SubElement(
                geom,
                "capsule",
                radius=f"{radius:.6f}",
                length=f"{max(1e-4, length):.6f}",
            )
    return len(prims)


def apply_tiangong3_capsule_collisions(
    spec,
    *,
    friction: tuple[float, float, float] = DEFAULT_FRICTION,
) -> dict[str, object]:
    """Replace mesh collisions on a Tiangong3 ``mujoco.MjSpec`` with calibrated capsules/spheres."""
    import mujoco

    paired: set[str] = {
        name for pair in spec.pairs for name in (pair.geomname1, pair.geomname2)
    }
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if geom.name in paired:
            continue
        if (geom.density == 0 or (geom.contype or geom.conaffinity)) and not (
            np.isfinite(geom.mass) and geom.mass > 0
        ):
            spec.delete(geom)

    referenced_meshes: set[str] = {
        geom.meshname
        for geom in spec.geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname
    }
    for mesh in list(spec.meshes):
        if mesh.name not in referenced_meshes:
            spec.delete(mesh)

    for body_name, prims in TIANGONG3_CAPSULE_BODY_PRIMITIVES.items():
        body = spec.body(body_name)
        for kind, geom_name, radius, p1, p2, condim in prims:
            if kind == "sphere":
                body.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[radius, 0.0, 0.0],
                    pos=list(p1),
                    group=1,
                    condim=condim,
                    friction=list(friction),
                    rgba=[0.752941, 0.752941, 0.752941, 1.0],
                )
            else:
                assert p2 is not None
                body.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[radius, 0.0, 0.0],
                    fromto=[*p1, *p2],
                    group=1,
                    condim=condim,
                    friction=list(friction),
                    rgba=[0.752941, 0.752941, 0.752941, 1.0],
                )

    existing_excludes: set[tuple[str, ...]] = {
        tuple(sorted((ex.bodyname1, ex.bodyname2))) for ex in spec.excludes
    }
    added_excludes: list[list[str]] = []
    for b1, b2 in TIANGONG3_CAPSULE_EXTRA_EXCLUDES:
        key = tuple(sorted((b1, b2)))
        if key not in existing_excludes:
            spec.add_exclude(bodyname1=b1, bodyname2=b2)
            existing_excludes.add(key)
            added_excludes.append([b1, b2])

    return {
        "profile": "capsule",
        "version": TIANGONG3_CAPSULE_PROFILE_VERSION,
        "collision_preserved": False,
        "mesh_collision_geoms": 0,
        "analytic_collision_geoms": len(list(spec.geoms)),
        "foot_collision_geoms": list(TIANGONG3_CAPSULE_FOOT_GEOM_NAMES),
        "removed_internal_bodies": list(TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES),
        "added_compound_joint_excludes": added_excludes,
    }
