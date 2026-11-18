"""Automated Mass, CoM, and Multi-Body 3D Forward Kinematics Parity Verifier (MuJoCo <-> Raisim2_FAST)."""
from __future__ import annotations

from typing import Sequence
import numpy as np

from .vec_engine import RaisimFastVecEngine


def verify_mjcf_raisim_parity(
    mj_model,
    engine: RaisimFastVecEngine,
    *,
    link_names: Sequence[str],
    entity_prefix: str = "",
    num_samples: int = 8,
    seed: int = 42,
) -> dict[str, float]:
    """Compare total mass, CoM, body positions, and body quaternions across random configurations."""
    import mujoco

    mj_data = mujoco.MjData(mj_model)
    rng = np.random.default_rng(seed)

    def _mj_bid(nm: str) -> int:
        if entity_prefix:
            try:
                return mj_model.body(f"{entity_prefix}{nm}").id
            except KeyError:
                pass
        return mj_model.body(nm).id

    bids = [_mj_bid(nm) for nm in link_names]
    subtree_root = bids[0]
    mj_total_mass = float(mj_model.body_subtreemass[subtree_root])

    max_mass_diff = 0.0
    max_com_diff = 0.0
    max_pos_diff = 0.0
    min_quat_abs_dot = 1.0

    for _ in range(num_samples):
        qpos = np.zeros((engine.num_envs, engine.nq), dtype=np.float32)
        qvel = np.zeros((engine.num_envs, engine.nv), dtype=np.float32)

        if engine.floating_base:
            qpos[:, 0:3] = rng.uniform(-0.5, 0.5, size=(engine.num_envs, 3))
            qpos[:, 2] += 1.0
            raw_q = rng.normal(size=(engine.num_envs, 4))
            raw_q /= np.linalg.norm(raw_q, axis=-1, keepdims=True)
            qpos[:, 3:7] = raw_q
            qpos[:, 7:] = rng.uniform(-0.35, 0.35, size=(engine.num_envs, engine.num_joints))
        else:
            qpos[:, :] = rng.uniform(-0.35, 0.35, size=(engine.num_envs, engine.num_joints))

        engine.set_all_states(qpos, qvel)
        rs_kin = engine.query_kinematics(0, link_names)

        mj_data.qpos[: engine.nq] = qpos[0].astype(np.float64)
        mj_data.qvel[: engine.nv] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        mj_com = np.asarray(mj_data.subtree_com[subtree_root], dtype=np.float64)
        mj_pos = np.asarray([mj_data.xpos[b] for b in bids], dtype=np.float64)
        mj_quat = np.asarray([mj_data.xquat[b] for b in bids], dtype=np.float64)

        max_mass_diff = max(max_mass_diff, abs(mj_total_mass - float(rs_kin["total_mass"])))
        max_com_diff = max(max_com_diff, float(np.max(np.abs(mj_com - rs_kin["com"]))))
        max_pos_diff = max(max_pos_diff, float(np.max(np.abs(mj_pos - rs_kin["body_pos"]))))
        dots = np.abs(np.sum(mj_quat * rs_kin["body_quat"], axis=-1))
        min_quat_abs_dot = min(min_quat_abs_dot, float(np.min(dots)))

    return {
        "mj_total_mass_kg": mj_total_mass,
        "max_mass_abs_diff_kg": max_mass_diff,
        "max_com_abs_diff_m": max_com_diff,
        "max_body_pos_abs_diff_m": max_pos_diff,
        "min_body_quat_abs_dot": min_quat_abs_dot,
        "num_links_checked": len(link_names),
        "num_random_poses": num_samples,
    }
