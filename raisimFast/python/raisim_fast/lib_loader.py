"""Automatic locator and on-demand C++20 compiler for `libraisim_fast_vec.so`."""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess


def get_raisim_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_lib_paths(repo_root: Path) -> list[Path]:
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    names = [f"libraisim_fast_vec{ext}", f"raisim_fast_vec{ext}"]
    candidates: list[Path] = []

    if "RAISIM_FAST_LIB" in os.environ:
        candidates.append(Path(os.environ["RAISIM_FAST_LIB"]))
    if "RAISIM_FAST_LIB_DIR" in os.environ:
        d = Path(os.environ["RAISIM_FAST_LIB_DIR"])
        candidates.extend(d / n for n in names)

    search_dirs = [
        repo_root / "build-py" / "raisimFast",
        repo_root / "build" / "raisimFast",
        repo_root / "build-release" / "raisimFast",
        repo_root / "install" / "lib",
        repo_root / "raisim" / "lib",
        repo_root / "raisim" / "linux" / "lib",
    ]
    for d in search_dirs:
        for n in names:
            candidates.append(d / n)
    return candidates


def _resolve_raisim_sdk_prefix(repo_root: Path) -> Path:
    if "RAISIM_PREFIX" in os.environ:
        p = Path(os.environ["RAISIM_PREFIX"])
        if (p / "include" / "raisim" / "World.hpp").is_file():
            return p
    for sub in ("raisim", "raisim/linux", "raisim/linux-arm", "raisim/mac", "raisim/m1", "install"):
        p = repo_root / sub
        if (p / "include" / "raisim" / "World.hpp").is_file():
            return p
    raise FileNotFoundError(
        f"Could not find RaiSim SDK headers under {repo_root}/raisim. "
        f"Run `./raisim_upgrade.sh` or `cmake -B build-py` first to fetch RaiSim 2.16+."
    )


def _resolve_eigen_include(repo_root: Path) -> Path:
    for env_key in ("EIGEN3_INCLUDE_DIR", "Eigen3_DIR"):
        if env_key in os.environ:
            p = Path(os.environ[env_key])
            if (p / "Eigen" / "Core").is_file():
                return p
            for parent in (p.parent, p.parent.parent, p.parent.parent.parent):
                if (parent / "include" / "eigen3" / "Eigen" / "Core").is_file():
                    return parent / "include" / "eigen3"
    sys_eigen = Path("/usr/include/eigen3")
    if (sys_eigen / "Eigen" / "Core").is_file():
        return sys_eigen
    bundled = repo_root / "thirdParty" / "Eigen3"
    if (bundled / "Eigen" / "Core").is_file():
        return bundled
    raise FileNotFoundError("Could not locate Eigen3 headers (set EIGEN3_INCLUDE_DIR or install libeigen3-dev).")


def build_raisim_fast_lib(output_path: Path | None = None) -> Path:
    """Compile `raisim_fast_vec.cpp` into a native OpenMP shared library."""
    repo_root = get_raisim_repo_root()
    cpp_src = repo_root / "raisimFast" / "cpp" / "raisim_fast_vec.cpp"
    if not cpp_src.is_file():
        raise FileNotFoundError(f"Missing C++ source: {cpp_src}")

    sdk_prefix = _resolve_raisim_sdk_prefix(repo_root)
    eigen_inc = _resolve_eigen_include(repo_root)
    lib_dir = sdk_prefix / "lib"
    openusd_dir = lib_dir / "openusd"

    if output_path is None:
        src_hash = hashlib.sha256(cpp_src.read_bytes()).hexdigest()[:12]
        cache_dir = Path(os.environ.get("RAISIM_FAST_CACHE_DIR", Path.home() / ".cache" / "raisim_fast"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        output_path = cache_dir / f"libraisim_fast_vec_{src_hash}{ext}"
        if output_path.is_file():
            return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cxx = os.environ.get("CXX", "g++")
    arch_flags = ["-march=native"] if platform.machine() in ("x86_64", "AMD64") else []
    rpath = f"{lib_dir}:{openusd_dir}" if openusd_dir.is_dir() else str(lib_dir)

    cmd = [
        cxx,
        "-std=gnu++20",
        "-O3",
        *arch_flags,
        "-fopenmp",
        "-shared",
        "-fPIC",
        str(cpp_src),
        f"-I{sdk_prefix / 'include'}",
        f"-I{eigen_inc}",
        f"-L{lib_dir}",
        f"-Wl,-rpath,{rpath}",
        "-lraisim",
        "-lpthread",
        "-ldl",
        "-o",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def load_raisim_fast_cdll(lib_path: str | Path | None = None) -> tuple[ctypes.CDLL, Path]:
    """Load `libraisim_fast_vec` via `ctypes.CDLL`, auto-compiling on first use if needed."""
    if lib_path is not None:
        resolved = Path(lib_path).resolve()
    else:
        repo_root = get_raisim_repo_root()
        found = next((p for p in _candidate_lib_paths(repo_root) if p.is_file()), None)
        resolved = found.resolve() if found is not None else build_raisim_fast_lib()

    cdll = ctypes.CDLL(str(resolved))
    cdll.rfast_create.restype = ctypes.c_void_p
    cdll.rtg3_create.restype = ctypes.c_void_p
    cdll.rfast_get_nq.restype = ctypes.c_int
    cdll.rfast_get_nv.restype = ctypes.c_int
    cdll.rfast_get_num_joints.restype = ctypes.c_int
    cdll.rfast_get_num_feet.restype = ctypes.c_int
    cdll.rfast_query_kinematics.restype = ctypes.c_int
    return cdll, resolved
