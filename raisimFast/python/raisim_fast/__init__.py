"""Raisim2_FAST: High-Throughput Vectorized OpenMP C++20 Engine & Analytical Primitive Compiler."""
from .lib_loader import build_raisim_fast_lib, get_raisim_repo_root, load_raisim_fast_cdll
from .mjcf_compiler import compile_mjcf_to_analytical_urdf, fit_analytical_primitive_from_aabb
from .parity import verify_mjcf_raisim_parity
from .tiangong_capsule_collision import (
    TIANGONG3_CAPSULE_EXTRA_EXCLUDES,
    TIANGONG3_CAPSULE_FOOT_GEOM_NAMES,
    TIANGONG3_CAPSULE_PROFILE_VERSION,
    TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES,
    apply_tiangong3_capsule_collisions,
    emit_tiangong3_capsule_urdf_collisions,
)
from .vec_engine import RaisimFastVecEngine

__version__ = "2.16.0"
__all__ = [
    "RaisimFastVecEngine",
    "TIANGONG3_CAPSULE_EXTRA_EXCLUDES",
    "TIANGONG3_CAPSULE_FOOT_GEOM_NAMES",
    "TIANGONG3_CAPSULE_PROFILE_VERSION",
    "TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES",
    "apply_tiangong3_capsule_collisions",
    "build_raisim_fast_lib",
    "compile_mjcf_to_analytical_urdf",
    "emit_tiangong3_capsule_urdf_collisions",
    "fit_analytical_primitive_from_aabb",
    "get_raisim_repo_root",
    "load_raisim_fast_cdll",
    "verify_mjcf_raisim_parity",
]
