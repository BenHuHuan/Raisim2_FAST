"""High-level NumPy & PyTorch vectorized wrapper around `libraisim_fast_vec`."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .lib_loader import load_raisim_fast_cdll


def _as_f32_ptr(arr: np.ndarray | None):
    if arr is None:
        return None
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class RaisimFastVecEngine:
    """OpenMP-vectorized RaiSim 2.16+ articulated multi-world engine."""

    def __init__(
        self,
        num_envs: int,
        urdf_template_path: str | Path,
        *,
        joint_names: Sequence[str],
        floating_base: bool = True,
        res_dir: str | Path | None = None,
        dt: float = 0.002,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        collision_excludes: Sequence[tuple[str, str]] = (),
        foot_body_names: Sequence[str] = (),
        anchor_body_name: str = "",
        armature: np.ndarray | None = None,
        damping: np.ndarray | None = None,
        frictionloss: np.ndarray | None = None,
        foot_friction: np.ndarray | None = None,
        anchor_delta_mass: np.ndarray | None = None,
        anchor_delta_com: np.ndarray | None = None,
        num_threads: int = 16,
        solver_max_iters: int = 25,
        solver_tol: float = 1e-6,
        lib_path: str | Path | None = None,
    ) -> None:
        self.lib, self.lib_path = load_raisim_fast_cdll(lib_path)
        self.num_envs = int(num_envs)
        self.num_threads = int(num_threads)
        self.floating_base = bool(floating_base)
        self.joint_names = list(joint_names)
        self.foot_body_names = list(foot_body_names)
        self.dt = float(dt)

        urdf_p = Path(urdf_template_path).resolve()
        res_p = Path(res_dir).resolve() if res_dir is not None else urdf_p.parent
        grav = np.ascontiguousarray(gravity, dtype=np.float32)

        nj = len(self.joint_names)

        def _prep_2d(x: np.ndarray | None) -> np.ndarray | None:
            if x is None:
                return None
            a = np.asarray(x, dtype=np.float32)
            if a.ndim == 1:
                a = np.broadcast_to(a[None, :], (self.num_envs, nj))
            return np.ascontiguousarray(a, dtype=np.float32)

        def _prep_1d(x: np.ndarray | None) -> np.ndarray | None:
            if x is None:
                return None
            a = np.asarray(x, dtype=np.float32)
            if a.ndim == 0:
                a = np.full((self.num_envs,), float(a), dtype=np.float32)
            return np.ascontiguousarray(a, dtype=np.float32)

        arm_c = _prep_2d(armature)
        dmp_c = _prep_2d(damping)
        frc_c = _prep_2d(frictionloss)
        fmu_c = _prep_1d(foot_friction)
        adm_c = _prep_1d(anchor_delta_mass)
        adc_c = (
            np.ascontiguousarray(anchor_delta_com, dtype=np.float32)
            if anchor_delta_com is not None
            else None
        )

        j_csv = ",".join(self.joint_names).encode()
        ex_csv = ";".join(f"{a}:{b}" for a, b in collision_excludes).encode()
        feet_csv = ",".join(self.foot_body_names).encode()
        anchor_b = (anchor_body_name or "").encode()

        self.handle = self.lib.rfast_create(
            int(self.num_envs),
            int(self.num_threads),
            int(1 if self.floating_base else 0),
            str(urdf_p).encode(),
            str(res_p).encode(),
            ctypes.c_double(self.dt),
            _as_f32_ptr(grav),
            j_csv,
            ex_csv,
            feet_csv,
            anchor_b,
            _as_f32_ptr(arm_c),
            _as_f32_ptr(dmp_c),
            _as_f32_ptr(frc_c),
            _as_f32_ptr(fmu_c),
            _as_f32_ptr(adm_c),
            _as_f32_ptr(adc_c),
            int(solver_max_iters),
            ctypes.c_double(solver_tol),
        )
        if not self.handle:
            raise RuntimeError(f"Failed to create RaisimFastVecEngine for {urdf_p}")

        self.nq = int(self.lib.rfast_get_nq(ctypes.c_void_p(self.handle)))
        self.nv = int(self.lib.rfast_get_nv(ctypes.c_void_p(self.handle)))
        self.num_joints = int(self.lib.rfast_get_num_joints(ctypes.c_void_p(self.handle)))
        self.num_feet = int(self.lib.rfast_get_num_feet(ctypes.c_void_p(self.handle)))

    @classmethod
    def from_asset_dir(
        cls,
        asset_dir: str | Path,
        num_envs: int,
        *,
        dt: float = 0.002,
        num_threads: int = 16,
        **kwargs,
    ) -> "RaisimFastVecEngine":
        asset_p = Path(asset_dir).resolve()
        meta_file = asset_p / "raisim_fast_meta.json"
        if not meta_file.is_file():
            meta_file = asset_p / "raisim_tg3_meta.json"
        meta = json.loads(meta_file.read_text())
        urdf_name = meta.get("template_urdf", "tiangong3_v2_29_template.urdf")
        return cls(
            num_envs=num_envs,
            urdf_template_path=asset_p / urdf_name,
            res_dir=asset_p,
            joint_names=meta["joint_names"],
            floating_base=meta.get("floating_base", True),
            dt=dt,
            collision_excludes=[tuple(p) for p in meta.get("collision_excludes", ())],
            foot_body_names=meta.get("foot_body_names", ()),
            anchor_body_name=meta.get("anchor_body_name", ""),
            armature=np.asarray(meta["nominal_armature"], dtype=np.float32) if "nominal_armature" in meta else None,
            damping=np.asarray(meta["nominal_damping"], dtype=np.float32) if "nominal_damping" in meta else None,
            frictionloss=np.asarray(meta["nominal_frictionloss"], dtype=np.float32) if "nominal_frictionloss" in meta else None,
            num_threads=num_threads,
            **kwargs,
        )

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.rfast_destroy(ctypes.c_void_p(self.handle))
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_all_states(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        qp = np.ascontiguousarray(qpos, dtype=np.float32)
        qv = np.ascontiguousarray(qvel, dtype=np.float32)
        assert qp.shape == (self.num_envs, self.nq)
        assert qv.shape == (self.num_envs, self.nv)
        self.lib.rfast_set_all_states(
            ctypes.c_void_p(self.handle),
            _as_f32_ptr(qp),
            _as_f32_ptr(qv),
        )

    def set_states_packed(self, env_ids: np.ndarray, qpos_packed: np.ndarray, qvel_packed: np.ndarray) -> None:
        ids = np.ascontiguousarray(env_ids, dtype=np.int32)
        k = int(ids.shape[0])
        if k == 0:
            return
        qp = np.ascontiguousarray(qpos_packed, dtype=np.float32)
        qv = np.ascontiguousarray(qvel_packed, dtype=np.float32)
        assert qp.shape == (k, self.nq)
        assert qv.shape == (k, self.nv)
        self.lib.rfast_set_states_packed(
            ctypes.c_void_p(self.handle),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            k,
            _as_f32_ptr(qp),
            _as_f32_ptr(qv),
        )

    def step_decimated(
        self,
        pos_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        effort_limit: np.ndarray,
        *,
        decimation: int = 10,
        root_wrench_w: np.ndarray | None = None,
        hist_len: int = 4,
    ) -> dict[str, np.ndarray]:
        tgt = np.ascontiguousarray(pos_target, dtype=np.float32)
        kp_a = np.ascontiguousarray(kp, dtype=np.float32)
        kd_a = np.ascontiguousarray(kd, dtype=np.float32)
        lim_a = np.ascontiguousarray(effort_limit, dtype=np.float32)
        wr_a = np.ascontiguousarray(root_wrench_w, dtype=np.float32) if root_wrench_w is not None else None

        qpos_out = np.empty((self.num_envs, self.nq), dtype=np.float32)
        qvel_out = np.empty((self.num_envs, self.nv), dtype=np.float32)
        qfrc_out = np.empty((self.num_envs, self.num_joints), dtype=np.float32)
        h = max(int(hist_len), 1)
        self_fnd = np.zeros((self.num_envs, h), dtype=np.float32)
        self_frc = np.zeros((self.num_envs, h, 3), dtype=np.float32)
        foot_fnd = np.zeros((self.num_envs, max(self.num_feet, 1)), dtype=np.float32)
        foot_frc = np.zeros((self.num_envs, max(self.num_feet, 1), 3), dtype=np.float32)

        self.lib.rfast_step_decimated(
            ctypes.c_void_p(self.handle),
            int(decimation),
            _as_f32_ptr(tgt),
            _as_f32_ptr(kp_a),
            _as_f32_ptr(kd_a),
            _as_f32_ptr(lim_a),
            _as_f32_ptr(wr_a),
            _as_f32_ptr(qpos_out),
            _as_f32_ptr(qvel_out),
            _as_f32_ptr(qfrc_out),
            _as_f32_ptr(self_fnd),
            _as_f32_ptr(self_frc),
            h,
            _as_f32_ptr(foot_fnd) if self.num_feet > 0 else None,
            _as_f32_ptr(foot_frc) if self.num_feet > 0 else None,
        )
        return {
            "qpos": qpos_out,
            "qvel": qvel_out,
            "qfrc_actuator": qfrc_out,
            "self_collision_found_hist": self_fnd,
            "self_collision_force_hist": self_frc,
            "foot_found": foot_fnd[:, : self.num_feet],
            "foot_force_w": foot_frc[:, : self.num_feet, :],
        }

    def query_kinematics(self, env_id: int, link_names: Sequence[str]) -> dict[str, np.ndarray | float]:
        tot_m = ctypes.c_double(0.0)
        com = np.zeros(3, dtype=np.float64)
        n = len(link_names)
        bpos = np.zeros((n, 3), dtype=np.float64)
        bquat = np.zeros((n, 4), dtype=np.float64)
        rc = self.lib.rfast_query_kinematics(
            ctypes.c_void_p(self.handle),
            int(env_id),
            ",".join(link_names).encode(),
            ctypes.byref(tot_m),
            com.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            bpos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            bquat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"rfast_query_kinematics failed with code {rc}")
        return {
            "total_mass": float(tot_m.value),
            "com": com,
            "body_pos": bpos,
            "body_quat": bquat,
        }
