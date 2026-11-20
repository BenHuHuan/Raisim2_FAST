# 天工（Tiangong3）训练碰撞体胶囊化优化：分析比对与结论报告（Raisim2_FAST + MLAB4_Fast 统一几何契约）

> [!IMPORTANT]
> **一句话核心结论：可行且收益显著，同时实现 GPU (`MLAB4_Fast`) 与 CPU (`Raisim2_FAST`) 跨仿真器几何零偏差对齐。**
> 通过采用 **Sim2Real 校准的全解析胶囊/球体碰撞体 (`tiangong3_capsule_v1`)** 替代天工原有的 39 个高面数 CAD 网格（或 64 顶点截断多面体），在 **100% 比特级保留全部刚体质量/惯量/关节电枢契约**、站立脚底间隙误差 **MAE = `0.273 mm`** 的前提下：
> - **GPU (`MLAB4_Fast` / MuJoCo 4.0 Warp) 物理与训练收益**：2048 环境下物理步提速 **`2.04×`**、峰值显存下降 **`78.0%`**（`5,750 -> 1,268 MiB`，8192 环境仅 `5,071 MiB`）；4096 地面起身工况比 `fast(hull64)` 快 **`+20.5%`**；真实 `SDK-A` 端到端训练（`--fp32-exact`）提速 **`+57.3%`**（30k 轮省 `6.6 小时`）。
> - **CPU (`Raisim2_FAST` / `compile_mjcf_to_analytical_urdf`) 原生几何同步**：`raisim_fast.mjcf_compiler` 在 `collision_profile="auto"` 下自动识别天工 29-DoF 骨架并直接生成完全一致的 **39 个 `$O(1)$` 解析胶囊/球体**（含精确 `1.35°` 脚底跟趾斜率 4 胶囊、手端 TCP 胶囊、剔除 14 个内部传感器冗余连杆，以及 17 对 2-hop 复合关节自碰撞排除），使 `2SIM_TRAIN` GPU+CPU 异构联合训练在两端共享完全同一套解析碰撞几何。
> - **闭环 Sim2Real 对齐验证**：将原全精度 CAD 训练收敛的 30k `TO19` 策略（`model_29999.pt`）零样本直接放入 `capsule` 环境做 128 世界 × 12s 闭环测试，**站立保持率（`96.88%`）与倒地起身转跟踪成功率（`98.44%`）与全精度 CAD `lean` 100.00% 完全一致**，关节跟踪 RMS 仅相差 `0.0049 rad`（`0.28°`），动作变化率更平滑（`-1.8%`）。

---

## 一、底层延迟根因诊断与跨仿真器收益

| 模型配置 | 活跃碰撞体数 | `mjGEOM_MESH` 数 | 解析原语数 (`capsule`/`sphere`) | 原始网格总面数 | 凸包总顶点数 (`hull_v`) | 凸包总面数 (`hull_f`) | 自碰撞排除对 `nexclude` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unitree G1 (`g1.xml`)** | 33 | **0** | **33** | 0 | 0 | 0 | 4 |
| **Tiangong3 `lean`（原 CAD 网格）** | 39 | **39** | 0 | 2,851,388 | **34,054** | **67,952** | 3 |
| **Tiangong3 `fast`（`maxhullvert=64`）** | 33 | **33** | 0 | 2,710,505 | **2,112** | **4,092** | 12 |
| **Tiangong3 `capsule` (`tiangong3_capsule_v1`)** | 39 | **0** | **39** | **0** | **0** | **0** | 20 |

1. **消除凸包图爬山与尖端多面体簇拥**：在 GPU `mujoco_warp` 中，`CAPSULE/SPHERE × PLANE` 为纯 `$O(1)$` 闭式解析解，彻底避开 `plane_convex` 5 轮邻接图爬山及 `convex_narrowphase` GJK/EPA/MultiCCD 重型缓冲；在 CPU `Raisim2_FAST` 中同样全链路走 `$O(1)$` 解析原语窄相（零三角网格/凸包 GJK 开销）。
2. **精确复刻脚底 `1.35°` 跟趾斜率与全身支撑链**：相比此前按 AABB 粗拟合的水平脚盒，每只脚用 **4 根纵向贴合胶囊**（`r=10 mm`）精确复刻 CAD 后跟 `z=-0.05677 m`、前掌 `z=-0.05305 m`、趾尖 `z=-0.05050 m` 的实测斜率与梯形支撑面，并剔除 14 个内部/传感器连杆的冗余碰撞体。
3. **补齐 17 对复合关节 2-hop 自碰撞排除**：消除标准 `TO19` 动作中 `pelvis <-> hip_roll` 的 `1,141` 次假自碰撞穿模惩罚，同时保留全部跨肢体真实自碰撞。

---

## 二、Raisim2_FAST 集成方式

- **核心模块**：[`raisimFast/python/raisim_fast/tiangong_capsule_collision.py`](../raisimFast/python/raisim_fast/tiangong_capsule_collision.py)
  - `emit_tiangong3_capsule_urdf_collisions(link_el, body_name)`：将 `tiangong3_capsule_v1` 解析胶囊/球体精确转换（MuJoCo `fromto` $\rightarrow$ URDF Z-aligned `origin(xyz, rpy)` + `length`）并写入 RaiSim URDF。
  - `apply_tiangong3_capsule_collisions(spec)`：提供与 `MLAB4_Fast` 完全同源的 `mujoco.MjSpec` 转换入口。
- **自动编译对齐**：[`compile_mjcf_to_analytical_urdf(..., collision_profile="auto")`](../raisimFast/python/raisim_fast/mjcf_compiler.py) 在检测到天工 29-DoF 骨架时默认启用 `tiangong3_capsule_v1` 并自动注入全部 17 对复合关节排除；传入 `collision_profile="aabb"` 可随时切回通用 AABB 拟合器。
- **自动化回归单测**：[`tests/test_raisim2_fast.py`](../tests/test_raisim2_fast.py)（覆盖天工胶囊 URDF 生成、39 解析原语数量、14 内部连杆零碰撞、17 对复合关节排除、与 `MLAB4_Fast` 几何一致及 RaiSim 物理步进验证）。

---

## 三、核心性能与精度对比汇总

### 1. 几何包络与接触精度审计（`TO19` 全量 5,729 帧对比 CAD `lean` 真值）

| 评估指标 | `lean`（39 CAD 网格真值） | `fast`（64 顶点凸包网格） | **`capsule` (`tiangong3_capsule_v1`)** |
| :--- | :---: | :---: | :---: |
| **动力学/惯量/关节契约一致性** | 基准 | 100% 比特级一致 | **100% 比特级一致** |
| **站立/行走地面间隙 MAE** (`n=3,979`) | `0.000 mm` | `0.102 mm` | **`0.273 mm`** (P50 `0.134 mm`, P95 `0.863 mm`) |
| **倒地起身地面间隙 MAE** (`n=1,750`) | `0.000 mm` | `0.381 mm` | **`2.973 mm`** (P50 `1.876 mm`, P95 `10.05 mm`) |
| **手端尖端单次触地簇拥接触点数** | `4.12` 个/次 | `2.68` 个/次 | **`2.02` 个/次** |
| **`TO19` 髋-盆复合关节假自碰撞次数** | `1,141` 次 | `0` 次 | **`0` 次** |

### 2. 并行扩展、显存与端到端训练收益（`tiangong3_v2_29_head_fixed`）

| 评估项目 | 原版 `lean` (CAD 网格) | `fast` (`maxhullvert=64`) | **`capsule` (`nmesh=0`)** | 相对 `lean` 收益 |
| :--- | :---: | :---: | :---: | :---: |
| **`MjSpec.compile()` 凸包构建 (`2048`)** | `1,018.99 ms` | `1,008.76 ms` | **`0.65 ms`** | **`1,568×` 提速** |
| **纯物理吞吐 (`2048` Mixed) / 物理显存** | `45,377 steps/s` / `5,750 MiB` | `75,649 steps/s` / `2,989 MiB` | **`92,626 steps/s` / `1,268 MiB`** | **`2.04×` / `-78.0%` 显存** |
| **纯物理吞吐 (`4096` Ground 起身)** | CUDA OOM | `86,004 steps/s` *(尖端退化)* | **`103,619 steps/s` / `2,536 MiB`** | **`+20.5%` vs `fast`** |
| **12 GiB 显存并行天花板** | `3,072` envs (`8,586 MiB`) | `6,144` envs (`8,816 MiB`) | **`8,192` envs (`5,071 MiB`)** | **`619 MiB / 1k envs`** (`1/4.54`) |
| **端到端 `SDK-A` 训练 (`2048 --fp32-exact`)** | `22,668 FPS` (`18.10 h` / 30k) | `31,345 FPS` (`13.08 h` / 30k) | **`35,657 FPS` (`11.49 h` / 30k)** | **`+57.3%`（30k 省 `6.6 h`）** |
| **30k `TO19` 策略零样本站立 / 起身成功率** | `96.88%` / `98.44%` | `96.88%` / `98.44%` | **`96.88%` / `98.44%`** | **`0.00%` 偏差（完全一致）** |
