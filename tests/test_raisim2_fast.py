"""End-to-end verification tests for Raisim2_FAST (`raisim_fast` + `libraisim_fast_vec`)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from raisim_fast import (
    RaisimFastVecEngine,
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
