"""End-to-end verification tests for Raisim2_FAST (`raisim_fast` + `libraisim_fast_vec`)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from raisim_fast import (
    RaisimFastVecEngine,
    TIANGONG3_CAPSULE_EXTRA_EXCLUDES,
    TIANGONG3_CAPSULE_PROFILE_VERSION,
    TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES,
    apply_tiangong3_capsule_collisions,
    compile_mjcf_to_analytical_urdf,
    load_raisim_fast_cdll,
    verify_mjcf_raisim_parity,
)


def _has_raisim_license() -> bool:
    lic = Path.home() / ".raisim" / "activation.raisim"
    return lic.is_file()


def test_cdll_loads_and_exports_symbols():
    cdll, path = load_raisim_fast_cdll()
    assert path.is_file()
    for sym in (
        "rfast_create",
        "rfast_destroy",
        "rfast_set_all_states",
        "rfast_set_states_packed",
        "rfast_step_decimated",
        "rfast_query_kinematics",
        "rtg3_create",
        "rtg3_step_decimated",
    ):
        assert hasattr(cdll, sym), f"Missing C ABI symbol: {sym}"


def test_mjcf_analytical_compiler(tmp_path: Path):
    mujoco = pytest.importorskip("mujoco")
    mjcf = """
    <mujoco model="biped_test">
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1"/>
        <body name="robot/pelvis" pos="0 0 0.85">
          <freejoint name="robot/root"/>
          <inertial pos="0 0 0.02" mass="12.5" diaginertia="0.12 0.10 0.08"/>
          <geom type="capsule" size="0.09 0.05"/>
          <body name="robot/left_thigh" pos="0 0.10 -0.10">
            <joint name="robot/hip_l" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <inertial pos="0 0 -0.15" mass="3.2" diaginertia="0.03 0.03 0.005"/>
            <geom type="capsule" size="0.045 0.14" pos="0 0 -0.15"/>
            <body name="robot/left_ankle_roll_link" pos="0 0 -0.32">
              <joint name="robot/ankle_l" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
              <inertial pos="0.02 0 -0.02" mass="0.8" diaginertia="0.002 0.004 0.004"/>
              <geom type="box" size="0.10 0.045 0.025" pos="0.02 0 -0.025"/>
            </body>
          </body>
          <body name="robot/right_thigh" pos="0 -0.10 -0.10">
            <joint name="robot/hip_r" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <inertial pos="0 0 -0.15" mass="3.2" diaginertia="0.03 0.03 0.005"/>
            <geom type="capsule" size="0.045 0.14" pos="0 0 -0.15"/>
            <body name="robot/right_ankle_roll_link" pos="0 0 -0.32">
              <joint name="robot/ankle_r" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
              <inertial pos="0.02 0 -0.02" mass="0.8" diaginertia="0.002 0.004 0.004"/>
              <geom type="box" size="0.10 0.045 0.025" pos="0.02 0 -0.025"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(mjcf)
    meta = compile_mjcf_to_analytical_urdf(m, tmp_path, robot_name="biped_test")
    assert meta["floating_base"] is True
    assert meta["joint_names"] == ["hip_l", "ankle_l", "hip_r", "ankle_r"]
    assert set(meta["foot_body_names"]) == {"left_ankle_roll_link", "right_ankle_roll_link"}
    assert (tmp_path / "robot_fast_template.urdf").is_file()

    if _has_raisim_license():
        eng = RaisimFastVecEngine.from_asset_dir(tmp_path, num_envs=64, dt=0.002, num_threads=4)
        try:
            links = [
                "pelvis",
                "left_thigh",
                "left_ankle_roll_link",
                "right_thigh",
                "right_ankle_roll_link",
            ]
            parity = verify_mjcf_raisim_parity(m, eng, link_names=links, entity_prefix="robot/")
            assert parity["max_mass_abs_diff_kg"] < 1e-5
            assert parity["max_com_abs_diff_m"] < 1e-6
            assert parity["max_body_pos_abs_diff_m"] < 1e-6
            assert parity["min_body_quat_abs_dot"] > 0.999999

            tgt = np.zeros((64, 4), dtype=np.float32)
            kp = np.full((64, 4), 80.0, dtype=np.float32)
            kd = np.full((64, 4), 4.0, dtype=np.float32)
            lim = np.full((64, 4), 120.0, dtype=np.float32)
            out = eng.step_decimated(tgt, kp, kd, lim, decimation=10)
            assert out["qpos"].shape == (64, 11)
            assert out["foot_force_w"].shape == (64, 2, 3)
            assert np.all(np.isfinite(out["qpos"]))

            # Test O(k) packed sparse reset
            ids = np.array([3, 17, 42], dtype=np.int32)
            qp_pack = np.zeros((3, 11), dtype=np.float32)
            qp_pack[:, 2] = 1.25
            qp_pack[:, 3] = 1.0
            qv_pack = np.zeros((3, 10), dtype=np.float32)
            eng.set_states_packed(ids, qp_pack, qv_pack)
            out2 = eng.step_decimated(tgt, kp, kd, lim, decimation=1)
            assert abs(float(out2["qpos"][17, 2]) - 1.25) < 0.01
        finally:
            eng.close()


_TIANGONG3_ALL_BODIES = (
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "imu_waist_link",
    "camera_body_front_link",
    "head_yaw_link",
    "head_pitch_link",
    "imu_head_link",
    "camera_head_link",
    "radar_head_link",
    "hip_pitch_l_link",
    "hip_roll_l_link",
    "hip_yaw_l_link",
    "knee_pitch_l_link",
    "ankle_pitch_l_link",
    "ankle_roll_l_link",
    "hip_pitch_r_link",
    "hip_roll_r_link",
    "hip_yaw_r_link",
    "knee_pitch_r_link",
    "ankle_pitch_r_link",
    "ankle_roll_r_link",
    "shoulder_pitch_l_link",
    "shoulder_roll_l_link",
    "shoulder_yaw_l_link",
    "elbow_pitch_l_link",
    "elbow_yaw_l_link",
    "wrist_pitch_l_link",
    "wrist_roll_l_link",
    "left_tcp_link",
    "shoulder_pitch_r_link",
    "shoulder_roll_r_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_r_link",
    "elbow_yaw_r_link",
    "wrist_pitch_r_link",
    "wrist_roll_r_link",
    "right_tcp_link",
)


def _make_synthetic_tiangong3_spec(mujoco):
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10.0, 10.0, 0.1], condim=3)
    cube_verts = [
        -0.02, -0.02, -0.02, 0.02, -0.02, -0.02, 0.02, 0.02, -0.02, -0.02, 0.02, -0.02,
        -0.02, -0.02, 0.02, 0.02, -0.02, 0.02, 0.02, 0.02, 0.02, -0.02, 0.02, 0.02,
    ]
    pelvis = spec.worldbody.add_body(
        name="pelvis", pos=[0.0, 0.0, 0.90], mass=5.2, inertia=[0.04, 0.04, 0.03], explicitinertial=1
    )
    pelvis.add_freejoint(name="floating_base")
    spec.add_mesh(name="pelvis_mesh", uservert=cube_verts)
    pelvis.add_geom(name="pelvis_mesh_col", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pelvis_mesh")
    for idx, body_name in enumerate(_TIANGONG3_ALL_BODIES[1:], start=1):
        b = pelvis.add_body(
            name=body_name,
            pos=[0.0, 0.0, -0.01 * idx],
            mass=0.5 + 0.05 * idx,
            inertia=[0.002, 0.002, 0.001],
            explicitinertial=1,
        )
        mesh_name = f"{body_name}_mesh"
        spec.add_mesh(name=mesh_name, uservert=cube_verts)
        b.add_geom(name=f"{body_name}_col", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name)
    return spec


def test_tiangong3_capsule_collision_compiler_and_spec(tmp_path: Path):
    import xml.etree.ElementTree as ET

    mujoco = pytest.importorskip("mujoco")

    # 1. MjSpec capsule converter eliminates all meshes and preserves explicit inertias
    base_model = _make_synthetic_tiangong3_spec(mujoco).compile()
    cap_spec = _make_synthetic_tiangong3_spec(mujoco)
    spec_meta = apply_tiangong3_capsule_collisions(cap_spec)
    cap_model = cap_spec.compile()

    assert spec_meta["version"] == TIANGONG3_CAPSULE_PROFILE_VERSION
    assert cap_model.nmesh == 0
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(base_model, field), getattr(cap_model, field))

    # 2. compile_mjcf_to_analytical_urdf auto-selects tiangong3_capsule_v1 for Tiangong3 skeletons
    out_auto = tmp_path / "tg3_auto"
    meta_auto = compile_mjcf_to_analytical_urdf(base_model, out_auto, robot_name="tiangong3_test")
    assert meta_auto["collision_profile"] == TIANGONG3_CAPSULE_PROFILE_VERSION
    assert meta_auto["total_analytical_primitives"] == 39

    urdf_root = ET.parse(out_auto / "robot_fast_nominal.urdf").getroot()
    link_map = {el.attrib["name"]: el for el in urdf_root.findall("link")}
    for internal_body in TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES:
        assert len(link_map[internal_body].findall("collision")) == 0
    assert len(link_map["ankle_roll_l_link"].findall("collision")) == 4
    assert len(link_map["ankle_roll_r_link"].findall("collision")) == 4

    exclude_set = {tuple(sorted(pair)) for pair in meta_auto["collision_excludes"]}
    for b1, b2 in TIANGONG3_CAPSULE_EXTRA_EXCLUDES:
        assert tuple(sorted((b1, b2))) in exclude_set

    # 3. Explicit collision_profile="aabb" still works as opt-out fallback
    out_aabb = tmp_path / "tg3_aabb"
    meta_aabb = compile_mjcf_to_analytical_urdf(
        base_model, out_aabb, robot_name="tiangong3_test", collision_profile="aabb"
    )
    assert meta_aabb["collision_profile"] == "aabb"
