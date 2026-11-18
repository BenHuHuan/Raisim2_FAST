#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" ]]; then
  __raisim_env_file="${BASH_SOURCE[0]}"
  if [[ "${__raisim_env_file}" == "$0" ]]; then
    echo "This script must be sourced: source /path/to/raisimLib/raisim_env.sh"
    exit 1
  fi
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  __raisim_env_file="${(%):-%x}"
else
  echo "This script must be sourced from Bash or Zsh."
  return 1 2>/dev/null || exit 1
fi

__raisim_env_dir="$(cd "$(dirname "${__raisim_env_file}")" && pwd)"
export RAISIM_DIR="${__raisim_env_dir}"

if [[ -z "${RAISIM_OS:-}" ]]; then
  case "$(uname -s)" in
    Linux)
      if [[ "$(uname -m)" == arm* || "$(uname -m)" == aarch64 ]]; then
        if [[ -d "${RAISIM_DIR}/raisim/linux-arm" ]]; then
          export RAISIM_OS="linux-arm"
        else
          export RAISIM_OS="linux"
        fi
      else
        export RAISIM_OS="linux"
      fi
      ;;
    Darwin)
      if [[ "$(uname -m)" == arm64 && -d "${RAISIM_DIR}/raisim/m1" ]]; then
        export RAISIM_OS="m1"
      else
        export RAISIM_OS="mac"
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      export RAISIM_OS="win32"
      ;;
    *)
      export RAISIM_OS="linux"
      ;;
  esac
fi

__raisim_lib_var="LD_LIBRARY_PATH"
if [[ "${RAISIM_OS}" == "mac" || "${RAISIM_OS}" == "m1" ]]; then
  __raisim_lib_var="DYLD_LIBRARY_PATH"
fi

__raisim_prepend_path() {
  local var_name="$1"
  local entry="$2"
  if [[ -z "${entry}" ]]; then
    return
  fi
  local current_value
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    current_value="${(P)var_name}"
  else
    current_value="${!var_name}"
  fi
  case ":${current_value}:" in
    *":${entry}:"*) return ;;
  esac
  if [[ -n "${current_value}" ]]; then
    export "${var_name}=${entry}:${current_value}"
  else
    export "${var_name}=${entry}"
  fi
  echo "Added to ${var_name}: ${entry}"
}

__raisim_prepend_package_lib() {
  local package_name="$1"
  local legacy_lib="${RAISIM_DIR}/${package_name}/${RAISIM_OS}/lib"
  local flat_lib="${RAISIM_DIR}/${package_name}/lib"

  if [[ -d "${legacy_lib}" ]]; then
    __raisim_prepend_path "${__raisim_lib_var}" "${legacy_lib}"
    if [[ -d "${legacy_lib}/openusd" ]]; then
      __raisim_prepend_path "${__raisim_lib_var}" "${legacy_lib}/openusd"
    fi
  elif [[ -d "${flat_lib}" ]]; then
    __raisim_prepend_path "${__raisim_lib_var}" "${flat_lib}"
    if [[ -d "${flat_lib}/openusd" ]]; then
      __raisim_prepend_path "${__raisim_lib_var}" "${flat_lib}/openusd"
    fi
  fi
}

__raisim_prepend_package_lib "raisim"
__raisim_prepend_package_lib "rayrai"

# Sanitize mismatched /opt/ros Python entries when active Python major.minor differs.
__raisim_python="$(command -v python3 || command -v python)"
if [[ -n "${__raisim_python}" && -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$("${__raisim_python}" - <<'EOF'
import os, sys
cur_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
kept = []
for entry in os.environ.get("PYTHONPATH", "").split(":"):
    if not entry:
        continue
    if "/opt/ros/" in entry and "python" in entry and cur_tag not in entry:
        continue
    kept.append(entry)
print(":".join(kept))
EOF
)"
fi

# raisimpy and raisim_fast built from this checkout.
__raisim_py_dir="${RAISIMPY_BUILD_DIR:-${RAISIM_DIR}/build-py/raisimPy}"
__raisim_prepend_path "PYTHONPATH" "${__raisim_py_dir}"
if [[ -d "${RAISIM_DIR}/raisimFast/python" ]]; then
  __raisim_prepend_path "PYTHONPATH" "${RAISIM_DIR}/raisimFast/python"
fi
export RAISIM_FAST_LIB_DIR="${RAISIM_FAST_LIB_DIR:-${RAISIM_DIR}/build-py/raisimFast}"
if [[ -n "${__raisim_python}" && -d "${__raisim_py_dir}" ]]; then
  "${__raisim_python}" - "${__raisim_py_dir}" <<'EOF'
import importlib.machinery, pathlib, sys
build = pathlib.Path(sys.argv[1])
built = sorted(p.name for p in build.glob("raisimpy.*") if p.suffix in (".so", ".pyd"))
if built and not any("raisimpy" + s in built for s in importlib.machinery.EXTENSION_SUFFIXES):
    caches = [c for c in (build / "CMakeCache.txt", build.parent / "CMakeCache.txt") if c.is_file()]
    exe = next((l.split("=", 1)[1] for c in caches for l in c.read_text().splitlines()
                if l.startswith("Python_EXECUTABLE:")), "the Python it was built with")
    print(f"Warning: {built[0]} cannot be imported by {sys.executable} (Python {sys.version.split()[0]}).")
    print(f"  Use {exe}, or rebuild with -DPython_EXECUTABLE={sys.executable}.")
EOF
fi

unset -f __raisim_prepend_package_lib
unset -f __raisim_prepend_path
unset __raisim_lib_var
unset __raisim_py_dir
unset __raisim_python
unset __raisim_env_dir
unset __raisim_env_file
