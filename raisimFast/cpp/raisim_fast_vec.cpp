// Raisim2_FAST: Robot-Agnostic Vectorized OpenMP C++20 Engine for RaiSim 2.16+.
// Supports arbitrary floating-base & fixed-base articulated robots (1..120 DoF),
// fused multi-substep explicit PD + torque clamping, per-world Domain Randomization,
// BisectionContactSolver tuning, O(k) packed sparse state resets, rolling self-collision
// history, multi-foot world-frame netforce, and exact MuJoCo <-> RaiSim frame conversion.

#include <raisim/World.hpp>
#include <raisim/object/ArticulatedSystem/ArticulatedSystem.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Stack-allocated Eigen vector up to 128 generalized coordinates (ZERO heap malloc in OpenMP loop!)
using StackVecXd = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 128, 1>;

std::vector<std::string> split_str(const std::string& s, char delim) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.length(), to);
    pos += to.length();
  }
}

size_t find_movable_body_exact(const raisim::ArticulatedSystem* art, const std::string& link_name) {
  const auto& bnames = art->getBodyNames();
  for (size_t i = 0; i < bnames.size(); ++i) {
    if (bnames[i] == link_name) return i;
  }
  return size_t(-1);
}

size_t resolve_movable_body_idx(const raisim::ArticulatedSystem* art, const std::string& link_name) {
  if (link_name.empty()) return size_t(-1);
  size_t idx = find_movable_body_exact(art, link_name);
  if (idx != size_t(-1)) return idx;
  size_t fidx = art->getFrameIdxByLinkName(link_name);
  if (fidx != size_t(-1)) {
    return art->getFrameByIdx(fidx).parentId;
  }
  return size_t(-1);
}

struct RaisimFastVec {
  int num_envs = 0;
  int num_threads = 1;
  bool floating_base = true;
  int num_joints = 0;
  int nq = 0;
  int nv = 0;
  int q_jnt_offset = 7;
  int v_jnt_offset = 6;
  double dt = 0.002;
  int solver_max_iters = 25;
  double solver_tol = 1e-6;

  std::string urdf_template;
  std::string res_dir;
  std::vector<std::string> joint_names;
  std::vector<size_t> foot_body_indices;
  size_t anchor_body_idx = size_t(-1);
  double base_anchor_mass = 0.0;
  Eigen::Vector3d base_anchor_com = Eigen::Vector3d::Zero();

  std::vector<std::unique_ptr<raisim::World>> worlds;
  std::vector<raisim::ArticulatedSystem*> robots;
  std::vector<raisim::Ground*> grounds;
};

inline void apply_mj_state_to_raisim(
    raisim::ArticulatedSystem* art,
    raisim::World* world,
    bool floating_base,
    int num_joints,
    int nq,
    int nv,
    const float* qp,
    const float* qv
) {
  StackVecXd gc(nq);
  StackVecXd gv(nv);

  if (floating_base) {
    gc[0] = qp[0];
    gc[1] = qp[1];
    gc[2] = qp[2];
    Eigen::Quaterniond q_wb(qp[3], qp[4], qp[5], qp[6]);
    if (q_wb.norm() < 1e-12) {
      q_wb = Eigen::Quaterniond::Identity();
    } else {
      q_wb.normalize();
    }
    gc[3] = q_wb.w();
    gc[4] = q_wb.x();
    gc[5] = q_wb.y();
    gc[6] = q_wb.z();
    for (int j = 0; j < num_joints; ++j) {
      gc[7 + j] = qp[7 + j];
    }

    // MuJoCo qvel[0..2] is world-frame v_w; qvel[3..5] is BODY-frame omega_b.
    // RaiSim gv[0..2] is world-frame v_w; gv[3..5] is WORLD-frame omega_w.
    gv[0] = qv[0];
    gv[1] = qv[1];
    gv[2] = qv[2];
    Eigen::Vector3d omega_b(qv[3], qv[4], qv[5]);
    Eigen::Vector3d omega_w = q_wb.toRotationMatrix() * omega_b;
    gv[3] = omega_w[0];
    gv[4] = omega_w[1];
    gv[5] = omega_w[2];
    for (int j = 0; j < num_joints; ++j) {
      gv[6 + j] = qv[6 + j];
    }
  } else {
    for (int j = 0; j < num_joints; ++j) {
      gc[j] = qp[j];
      gv[j] = qv[j];
    }
  }

  art->setState(gc, gv);
  world->resetContactSolverWarmStart();
}

}  // namespace

extern "C" {

void* rfast_create(
    int num_envs,
    int num_threads,
    int floating_base,
    const char* urdf_template_path,
    const char* res_dir_cstr,
    double dt,
    const float* gravity_xyz,
    const char* joint_names_csv,
    const char* exclude_pairs_csv,
    const char* foot_links_csv,
    const char* anchor_link_cstr,
    const float* armature_f32,          // [num_envs, num_joints] or nullptr
    const float* damping_f32,           // [num_envs, num_joints] or nullptr
    const float* frictionloss_f32,      // [num_envs, num_joints] or nullptr
    const float* foot_friction_f32,     // [num_envs] or nullptr
    const float* anchor_delta_mass_f32, // [num_envs] or nullptr
    const float* anchor_delta_com_f32,  // [num_envs, 3] or nullptr
    int solver_max_iters,
    double solver_tol
) {
  try {
    auto* vec = new RaisimFastVec();
    vec->num_envs = num_envs;
    vec->num_threads = std::max(1, num_threads);
    vec->floating_base = (floating_base != 0);
    vec->dt = dt;
    vec->solver_max_iters = solver_max_iters > 0 ? solver_max_iters : 25;
    vec->solver_tol = solver_tol > 0.0 ? solver_tol : 1e-6;
    vec->res_dir = res_dir_cstr ? res_dir_cstr : "";
    vec->joint_names = split_str(joint_names_csv ? joint_names_csv : "", ',');
    vec->num_joints = static_cast<int>(vec->joint_names.size());
    vec->q_jnt_offset = vec->floating_base ? 7 : 0;
    vec->v_jnt_offset = vec->floating_base ? 6 : 0;
    vec->nq = vec->q_jnt_offset + vec->num_joints;
    vec->nv = vec->v_jnt_offset + vec->num_joints;

    if (vec->nq <= 0 || vec->nq > 128) {
      delete vec;
      throw std::runtime_error("Unsupported generalized coordinate dimension nq=" + std::to_string(vec->nq));
    }

    std::ifstream ifs(urdf_template_path);
    if (!ifs) {
      delete vec;
      throw std::runtime_error(std::string("Cannot open URDF template: ") + urdf_template_path);
    }
    vec->urdf_template.assign(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>()
    );

    std::vector<std::pair<std::string, std::string>> ex_names;
    for (const auto& tok : split_str(exclude_pairs_csv ? exclude_pairs_csv : "", ';')) {
      auto ab = split_str(tok, ':');
      if (ab.size() == 2) ex_names.emplace_back(ab[0], ab[1]);
    }
    std::vector<std::string> foot_names = split_str(foot_links_csv ? foot_links_csv : "", ',');
    std::string anchor_link = anchor_link_cstr ? anchor_link_cstr : "";

    vec->worlds.resize(num_envs);
    vec->robots.resize(num_envs, nullptr);
    vec->grounds.resize(num_envs, nullptr);

    Eigen::Vector3d grav(gravity_xyz[0], gravity_xyz[1], gravity_xyz[2]);
    std::vector<std::pair<size_t, size_t>> ex_body_indices;

    auto init_one_env = [&](int i) {
      std::string urdf_str = vec->urdf_template;
      char buf[64];
      for (int j = 0; j < vec->num_joints; ++j) {
        float damp = damping_f32 ? damping_f32[i * vec->num_joints + j] : 0.0f;
        float fric = frictionloss_f32 ? frictionloss_f32[i * vec->num_joints + j] : 0.0f;
        std::snprintf(buf, sizeof(buf), "%.8f", damp);
        replace_all(urdf_str, "__DAMP_" + std::to_string(j) + "__", buf);
        std::snprintf(buf, sizeof(buf), "%.8f", fric);
        replace_all(urdf_str, "__FRIC_" + std::to_string(j) + "__", buf);
      }

      auto world = std::make_unique<raisim::World>();
      world->setTimeStep(dt);
      world->setGravity(grav);
      world->setERP(0.0, 0.0);
      auto& solver_cfg = world->getContactSolver().getConfig();
      solver_cfg.error_to_terminate = vec->solver_tol;
      solver_cfg.maxIteration = vec->solver_max_iters;

      auto* ground = world->addGround(0.0, "ground");
      raisim::ArticulatedSystemOption opt;
      opt.doNotCollideWithParent = true;
      opt.convexifyCollisionMeshes = false;
      auto* art = world->addArticulatedSystem(
          urdf_str,
          vec->res_dir,
          vec->joint_names,
          raisim::CollisionGroup(1),
          raisim::CollisionGroup(-1),
          opt
      );
      art->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

      if (i == 0) {
        if (static_cast<int>(art->getGeneralizedCoordinateDim()) != vec->nq ||
            static_cast<int>(art->getDOF()) != vec->nv) {
          throw std::runtime_error(
              "Dimension mismatch: expected nq=" + std::to_string(vec->nq) +
              " nv=" + std::to_string(vec->nv) + " got nq=" +
              std::to_string(art->getGeneralizedCoordinateDim()) +
              " nv=" + std::to_string(art->getDOF())
          );
        }
        for (const auto& fn : foot_names) {
          size_t fidx = resolve_movable_body_idx(art, fn);
          if (fidx == size_t(-1)) {
            throw std::runtime_error("Cannot resolve foot body: " + fn);
          }
          vec->foot_body_indices.push_back(fidx);
        }
        if (!anchor_link.empty()) {
          vec->anchor_body_idx = resolve_movable_body_idx(art, anchor_link);
          if (vec->anchor_body_idx != size_t(-1)) {
            vec->base_anchor_mass = art->getMass(vec->anchor_body_idx);
            vec->base_anchor_com = art->getBodyCOM_B()[vec->anchor_body_idx].e();
          }
        }
        for (const auto& pr : ex_names) {
          size_t b1 = resolve_movable_body_idx(art, pr.first);
          size_t b2 = resolve_movable_body_idx(art, pr.second);
          if (b1 != size_t(-1) && b2 != size_t(-1) && b1 != b2) {
            ex_body_indices.emplace_back(b1, b2);
          }
        }
      }

      for (const auto& pr : ex_body_indices) {
        art->ignoreCollisionBetween(pr.first, pr.second);
      }
      art->updateSelfCollisionCache(*world);

      if (armature_f32) {
        raisim::VecDyn rotor(vec->nv);
        rotor.setZero();
        for (int j = 0; j < vec->num_joints; ++j) {
          rotor[vec->v_jnt_offset + j] = std::max(0.0, double(armature_f32[i * vec->num_joints + j]));
        }
        art->setRotorInertia(rotor);
      }

      if (damping_f32) {
        raisim::VecDyn damp_vec(vec->nv);
        damp_vec.setZero();
        for (int j = 0; j < vec->num_joints; ++j) {
          damp_vec[vec->v_jnt_offset + j] = std::max(0.0, double(damping_f32[i * vec->num_joints + j]));
        }
        art->setJointDamping(damp_vec);
      }

      if ((anchor_delta_mass_f32 || anchor_delta_com_f32) && vec->anchor_body_idx != size_t(-1)) {
        if (anchor_delta_mass_f32) {
          double dm = double(anchor_delta_mass_f32[i]);
          art->getMass()[vec->anchor_body_idx] = std::max(0.05, vec->base_anchor_mass + dm);
        }
        if (anchor_delta_com_f32) {
          raisim::Vec<3> new_com;
          new_com[0] = vec->base_anchor_com[0] + double(anchor_delta_com_f32[i * 3 + 0]);
          new_com[1] = vec->base_anchor_com[1] + double(anchor_delta_com_f32[i * 3 + 1]);
          new_com[2] = vec->base_anchor_com[2] + double(anchor_delta_com_f32[i * 3 + 2]);
          art->getBodyCOM_B()[vec->anchor_body_idx] = new_com;
        }
        art->updateMassInfo();
      }

      double mu_foot = foot_friction_f32 ? std::max(0.01, double(foot_friction_f32[i])) : 1.0;
      world->setMaterialPairProp("default", "ground", 1.0, 0.0, 0.01);
      world->setMaterialPairProp("foot", "ground", mu_foot, 0.0, 0.01);
      if (!vec->foot_body_indices.empty()) {
        for (auto& col : art->getCollisionBodies()) {
          for (size_t fidx : vec->foot_body_indices) {
            if (col.localIdx == fidx) {
              col.setMaterial("foot");
              break;
            }
          }
        }
      }

      vec->grounds[i] = ground;
      vec->robots[i] = art;
      vec->worlds[i] = std::move(world);
    };

    init_one_env(0);

    std::string parallel_err;
    #pragma omp parallel for num_threads(vec->num_threads) schedule(static)
    for (int i = 1; i < num_envs; ++i) {
      try {
        init_one_env(i);
      } catch (const std::exception& e) {
        #pragma omp critical
        {
          if (parallel_err.empty()) parallel_err = e.what();
        }
      }
    }
    if (!parallel_err.empty()) {
      delete vec;
      throw std::runtime_error(parallel_err);
    }

    return static_cast<void*>(vec);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rfast_create ERROR] %s\n", e.what());
    return nullptr;
  }
}

void rfast_destroy(void* handle) {
  if (!handle) return;
  delete static_cast<RaisimFastVec*>(handle);
}

int rfast_get_nq(void* handle) { return static_cast<RaisimFastVec*>(handle)->nq; }
int rfast_get_nv(void* handle) { return static_cast<RaisimFastVec*>(handle)->nv; }
int rfast_get_num_joints(void* handle) { return static_cast<RaisimFastVec*>(handle)->num_joints; }
int rfast_get_num_feet(void* handle) { return static_cast<int>(static_cast<RaisimFastVec*>(handle)->foot_body_indices.size()); }

void rfast_set_all_states(void* handle, const float* qpos_f32, const float* qvel_f32) {
  auto* vec = static_cast<RaisimFastVec*>(handle);
  const int nq = vec->nq;
  const int nv = vec->nv;
  #pragma omp parallel for num_threads(vec->num_threads) schedule(dynamic, 8)
  for (int i = 0; i < vec->num_envs; ++i) {
    apply_mj_state_to_raisim(
        vec->robots[i],
        vec->worlds[i].get(),
        vec->floating_base,
        vec->num_joints,
        nq,
        nv,
        qpos_f32 + static_cast<size_t>(i) * nq,
        qvel_f32 + static_cast<size_t>(i) * nv
    );
  }
}

void rfast_set_states_subset(
    void* handle,
    const int32_t* env_ids_i32,
    int count,
    const float* qpos_f32,
    const float* qvel_f32
) {
  auto* vec = static_cast<RaisimFastVec*>(handle);
  const int nq = vec->nq;
  const int nv = vec->nv;
  #pragma omp parallel for num_threads(vec->num_threads) schedule(static) if(count >= 64)
  for (int k = 0; k < count; ++k) {
    int i = env_ids_i32[k];
    apply_mj_state_to_raisim(
        vec->robots[i],
        vec->worlds[i].get(),
        vec->floating_base,
        vec->num_joints,
        nq,
        nv,
        qpos_f32 + static_cast<size_t>(i) * nq,
        qvel_f32 + static_cast<size_t>(i) * nv
    );
  }
}

void rfast_set_states_packed(
    void* handle,
    const int32_t* env_ids_i32,
    int count,
    const float* qpos_packed_f32,
    const float* qvel_packed_f32
) {
  auto* vec = static_cast<RaisimFastVec*>(handle);
  const int nq = vec->nq;
  const int nv = vec->nv;
  #pragma omp parallel for num_threads(vec->num_threads) schedule(static) if(count >= 64)
  for (int k = 0; k < count; ++k) {
    int i = env_ids_i32[k];
    apply_mj_state_to_raisim(
        vec->robots[i],
        vec->worlds[i].get(),
        vec->floating_base,
        vec->num_joints,
        nq,
        nv,
        qpos_packed_f32 + static_cast<size_t>(k) * nq,
        qvel_packed_f32 + static_cast<size_t>(k) * nv
    );
  }
}

void rfast_step_decimated(
    void* handle,
    int decimation,
    const float* pos_target_f32,      // [num_envs, num_joints]
    const float* kp_f32,              // [num_envs, num_joints]
    const float* kd_f32,              // [num_envs, num_joints]
    const float* effort_limit_f32,    // [num_envs, num_joints]
    const float* root_wrench_w_f32,   // [num_envs, 6] optional world force+torque on root (or nullptr)
    float* qpos_out_f32,              // [num_envs, nq]
    float* qvel_out_f32,              // [num_envs, nv]
    float* qfrc_act_out_f32,          // [num_envs, num_joints]
    float* self_found_hist_f32,       // [num_envs, hist_len] (or nullptr)
    float* self_force_hist_f32,       // [num_envs, hist_len, 3] (or nullptr)
    int hist_len,
    float* foot_found_out_f32,        // [num_envs, num_feet] (or nullptr)
    float* foot_force_out_f32         // [num_envs, num_feet, 3] (or nullptr)
) {
  auto* vec = static_cast<RaisimFastVec*>(handle);
  const double inv_dt = 1.0 / vec->dt;
  const int nj = vec->num_joints;
  const int nq = vec->nq;
  const int nv = vec->nv;
  const int q_off = vec->q_jnt_offset;
  const int v_off = vec->v_jnt_offset;
  const bool floating = vec->floating_base;
  const int n_feet = static_cast<int>(vec->foot_body_indices.size());
  const size_t* foot_indices = vec->foot_body_indices.data();
  const int h_cap = std::max(hist_len, 0);

  #pragma omp parallel for num_threads(vec->num_threads) schedule(dynamic, 8)
  for (int i = 0; i < vec->num_envs; ++i) {
    auto* world = vec->worlds[i].get();
    auto* art = vec->robots[i];
    size_t art_obj_idx = art->getIndexInWorld();

    const float* tgt = pos_target_f32 + static_cast<size_t>(i) * nj;
    const float* kp = kp_f32 + static_cast<size_t>(i) * nj;
    const float* kd = kd_f32 + static_cast<size_t>(i) * nj;
    const float* lim = effort_limit_f32 + static_cast<size_t>(i) * nj;
    const float* wr = root_wrench_w_f32 ? (root_wrench_w_f32 + static_cast<size_t>(i) * 6) : nullptr;

    float* self_fnd = self_found_hist_f32 ? (self_found_hist_f32 + static_cast<size_t>(i) * h_cap) : nullptr;
    float* self_frc = self_force_hist_f32 ? (self_force_hist_f32 + static_cast<size_t>(i) * h_cap * 3) : nullptr;
    if (self_fnd && self_frc) {
      for (int h = 0; h < h_cap; ++h) {
        self_fnd[h] = 0.0f;
        self_frc[h * 3 + 0] = 0.0f;
        self_frc[h * 3 + 1] = 0.0f;
        self_frc[h * 3 + 2] = 0.0f;
      }
    }

    StackVecXd tau(nv);
    tau.setZero();

    for (int sub = 0; sub < decimation; ++sub) {
      const auto& gc = art->getGeneralizedCoordinate();
      const auto& gv = art->getGeneralizedVelocity();

      for (int j = 0; j < nj; ++j) {
        double pos_err = double(tgt[j]) - gc[q_off + j];
        double vel_err = -gv[v_off + j];
        double t = double(kp[j]) * pos_err + double(kd[j]) * vel_err;
        double l = double(lim[j]);
        if (t > l) t = l;
        else if (t < -l) t = -l;
        tau[v_off + j] = t;
      }
      art->setGeneralizedForce(tau);

      if (wr) {
        raisim::Vec<3> f_ext = {double(wr[0]), double(wr[1]), double(wr[2])};
        raisim::Vec<3> t_ext = {double(wr[3]), double(wr[4]), double(wr[5])};
        if (f_ext.norm() > 1e-9) {
          art->setExternalForce(0, raisim::ArticulatedSystem::Frame::WORLD_FRAME, f_ext,
                                raisim::ArticulatedSystem::Frame::BODY_FRAME, {0.0, 0.0, 0.0});
        }
        if (t_ext.norm() > 1e-9) {
          art->setExternalTorque(0, t_ext);
        }
      }

      world->integrate();

      int h_slot = (decimation - 1) - sub;
      if (self_fnd && self_frc && h_slot >= 0 && h_slot < h_cap) {
        double max_norm2 = 0.0;
        Eigen::Vector3d best_f = Eigen::Vector3d::Zero();
        int self_cnt = 0;

        for (const auto& c : art->getContacts()) {
          if (c.skip()) continue;
          if (!c.isSelfCollision() && c.getPairObjectIndex() != art_obj_idx) continue;
          if (!c.isObjectA()) continue;
          Eigen::Vector3d imp_c = c.getImpulse().e();
          Eigen::Vector3d f_c = imp_c * inv_dt;
          double n2 = f_c.squaredNorm();
          self_cnt += 1;
          if (n2 > max_norm2) {
            max_norm2 = n2;
            best_f = f_c;
          }
        }
        self_fnd[h_slot] = float(self_cnt);
        self_frc[h_slot * 3 + 0] = float(best_f[0]);
        self_frc[h_slot * 3 + 1] = float(best_f[1]);
        self_frc[h_slot * 3 + 2] = float(best_f[2]);
      }
    }

    const auto& gc = art->getGeneralizedCoordinate();
    const auto& gv = art->getGeneralizedVelocity();

    float* qp_out = qpos_out_f32 + static_cast<size_t>(i) * nq;
    float* qv_out = qvel_out_f32 + static_cast<size_t>(i) * nv;
    float* qf_out = qfrc_act_out_f32 + static_cast<size_t>(i) * nj;

    if (floating) {
      qp_out[0] = float(gc[0]);
      qp_out[1] = float(gc[1]);
      qp_out[2] = float(gc[2]);
      Eigen::Quaterniond q_wb(gc[3], gc[4], gc[5], gc[6]);
      q_wb.normalize();
      qp_out[3] = float(q_wb.w());
      qp_out[4] = float(q_wb.x());
      qp_out[5] = float(q_wb.y());
      qp_out[6] = float(q_wb.z());
      for (int j = 0; j < nj; ++j) qp_out[7 + j] = float(gc[7 + j]);

      qv_out[0] = float(gv[0]);
      qv_out[1] = float(gv[1]);
      qv_out[2] = float(gv[2]);
      Eigen::Vector3d omega_w(gv[3], gv[4], gv[5]);
      Eigen::Vector3d omega_b = q_wb.conjugate().toRotationMatrix() * omega_w;
      qv_out[3] = float(omega_b[0]);
      qv_out[4] = float(omega_b[1]);
      qv_out[5] = float(omega_b[2]);
      for (int j = 0; j < nj; ++j) qv_out[6 + j] = float(gv[6 + j]);
    } else {
      for (int j = 0; j < nj; ++j) {
        qp_out[j] = float(gc[j]);
        qv_out[j] = float(gv[j]);
      }
    }

    for (int j = 0; j < nj; ++j) {
      qf_out[j] = float(tau[v_off + j]);
    }

    if (n_feet > 0 && foot_found_out_f32 && foot_force_out_f32) {
      float* ft_fnd = foot_found_out_f32 + static_cast<size_t>(i) * n_feet;
      float* ft_frc = foot_force_out_f32 + static_cast<size_t>(i) * n_feet * 3;
      for (int f = 0; f < n_feet; ++f) {
        ft_fnd[f] = 0.0f;
        ft_frc[f * 3 + 0] = 0.0f;
        ft_frc[f * 3 + 1] = 0.0f;
        ft_frc[f * 3 + 2] = 0.0f;
      }

      for (const auto& c : art->getContacts()) {
        if (c.skip() || c.isSelfCollision()) continue;
        size_t bidx = c.getlocalBodyIndex();
        int matched_f = -1;
        for (int f = 0; f < n_feet; ++f) {
          if (bidx == foot_indices[f]) {
            matched_f = f;
            break;
          }
        }
        if (matched_f < 0) continue;
        Eigen::Vector3d imp_c = c.getImpulse().e();
        Eigen::Vector3d f_w = (c.getContactFrame().e().transpose() * imp_c) * inv_dt;
        if (!c.isObjectA()) f_w = -f_w;
        ft_fnd[matched_f] += 1.0f;
        ft_frc[matched_f * 3 + 0] += float(f_w[0]);
        ft_frc[matched_f * 3 + 1] += float(f_w[1]);
        ft_frc[matched_f * 3 + 2] += float(f_w[2]);
      }
    }
  }
}

int rfast_query_kinematics(
    void* handle,
    int env_id,
    const char* link_names_csv,
    double* total_mass_out,
    double* com_xyz_out,
    double* body_pos_out,
    double* body_quat_out
) {
  try {
    auto* vec = static_cast<RaisimFastVec*>(handle);
    auto* art = vec->robots.at(env_id);

    double tot_m = 0.0;
    for (double m : art->getMass()) tot_m += m;
    *total_mass_out = tot_m;

    auto com = art->getCOM();
    com_xyz_out[0] = com[0];
    com_xyz_out[1] = com[1];
    com_xyz_out[2] = com[2];

    auto names = split_str(link_names_csv ? link_names_csv : "", ',');
    for (size_t k = 0; k < names.size(); ++k) {
      const auto& nm = names[k];
      raisim::Vec<3> pos;
      raisim::Mat<3, 3> rot;
      size_t bidx = find_movable_body_exact(art, nm);
      if (bidx != size_t(-1)) {
        art->getPosition(bidx, pos);
        art->getOrientation(bidx, rot);
      } else {
        size_t fidx = art->getFrameIdxByLinkName(nm);
        if (fidx == size_t(-1)) throw std::runtime_error("Unknown link: " + nm);
        art->getFramePosition(fidx, pos);
        art->getFrameOrientation(fidx, rot);
      }
      body_pos_out[k * 3 + 0] = pos[0];
      body_pos_out[k * 3 + 1] = pos[1];
      body_pos_out[k * 3 + 2] = pos[2];
      Eigen::Quaterniond q(rot.e());
      q.normalize();
      if (q.w() < 0) q.coeffs() = -q.coeffs();
      body_quat_out[k * 4 + 0] = q.w();
      body_quat_out[k * 4 + 1] = q.x();
      body_quat_out[k * 4 + 2] = q.y();
      body_quat_out[k * 4 + 3] = q.z();
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rfast_query_kinematics ERROR] %s\n", e.what());
    return -1;
  }
}

// ============================================================================
// Backward-compatible rtg3_* symbols forwarding directly to rfast_*
// ============================================================================
void* rtg3_create(
    int num_envs,
    int num_threads,
    const char* urdf_template_path,
    const char* res_dir_cstr,
    double dt,
    const float* gravity_xyz,
    const char* joint_names_csv,
    const char* exclude_pairs_csv,
    const char* left_foot_link,
    const char* right_foot_link,
    const char* anchor_link,
    const float* armature_f32,
    const float* damping_f32,
    const float* frictionloss_f32,
    const float* foot_friction_f32,
    const float* anchor_delta_mass_f32,
    const float* anchor_delta_com_f32
) {
  std::string feet_csv;
  if (left_foot_link && *left_foot_link) feet_csv += left_foot_link;
  if (right_foot_link && *right_foot_link) {
    if (!feet_csv.empty()) feet_csv += ",";
    feet_csv += right_foot_link;
  }
  return rfast_create(
      num_envs, num_threads, 1, urdf_template_path, res_dir_cstr, dt, gravity_xyz,
      joint_names_csv, exclude_pairs_csv, feet_csv.c_str(), anchor_link,
      armature_f32, damping_f32, frictionloss_f32, foot_friction_f32,
      anchor_delta_mass_f32, anchor_delta_com_f32, 25, 1e-6
  );
}

void rtg3_destroy(void* handle) { rfast_destroy(handle); }

void rtg3_set_all_states(void* handle, const float* qpos_f32, const float* qvel_f32) {
  rfast_set_all_states(handle, qpos_f32, qvel_f32);
}

void rtg3_set_states_subset(void* handle, const int32_t* env_ids_i32, int count, const float* qpos_f32, const float* qvel_f32) {
  rfast_set_states_subset(handle, env_ids_i32, count, qpos_f32, qvel_f32);
}

void rtg3_set_states_packed(void* handle, const int32_t* env_ids_i32, int count, const float* qpos_packed_f32, const float* qvel_packed_f32) {
  rfast_set_states_packed(handle, env_ids_i32, count, qpos_packed_f32, qvel_packed_f32);
}

void rtg3_step_decimated(
    void* handle, int decimation,
    const float* pos_target_f32, const float* kp_f32, const float* kd_f32, const float* effort_limit_f32,
    const float* root_wrench_w_f32,
    float* qpos_out_f32, float* qvel_out_f32, float* qfrc_act_out_f32,
    float* self_found_hist_f32, float* self_force_hist_f32, int hist_len,
    float* foot_found_out_f32, float* foot_force_out_f32
) {
  rfast_step_decimated(
      handle, decimation, pos_target_f32, kp_f32, kd_f32, effort_limit_f32, root_wrench_w_f32,
      qpos_out_f32, qvel_out_f32, qfrc_act_out_f32,
      self_found_hist_f32, self_force_hist_f32, hist_len,
      foot_found_out_f32, foot_force_out_f32
  );
}

int rtg3_query_kinematics(
    void* handle, int env_id, const char* link_names_csv,
    double* total_mass_out, double* com_xyz_out, double* body_pos_out, double* body_quat_out
) {
  return rfast_query_kinematics(handle, env_id, link_names_csv, total_mass_out, com_xyz_out, body_pos_out, body_quat_out);
}

}  // extern "C"
