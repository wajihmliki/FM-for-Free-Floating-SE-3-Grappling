# Momentum-Projected Generative Warm-Starting for Free-Floating SE(3) Grappling

Code for the paper

> **Momentum-Projected Generative Warm-Starting for Free-Floating SE(3) Grappling**
> Wajih Imliki, Mohammad Mahdi Rahimi, Han-Lim Choi.
> Submitted to ICRA 2027 (under review).

A free-floating spacecraft with a 6-DOF arm must capture a tumbling target. Because the
base is unactuated, arm motion makes the base recoil through conservation of momentum.
A conditional flow-matching model generates only the arm's **joint trajectory**; the
six-DOF base pose is **reconstructed from the system momentum map**, so every warm start
conserves momentum by construction (Prop. 1) and its terminal reach error carries no
recoil term under exact execution (Prop. 2). Every candidate is certified by full
nonlinear forward integration against a 5 cm / 3° SE(3) grasp gate and validated in
MuJoCo.

Pipeline: grasp scenario → conditional flow matching → generated joint trajectory θ(t)
→ momentum-based base reconstruction → SE(3) trajectory optimizer → full nonlinear
verification → certified grasp.

## Installation

Tested with Python 3.10 and PyTorch 2.x (CUDA optional; several scripts default to
`--device cuda`).

```bash
pip install -r requirements.txt
```

Scripts are run from the repository root; datasets and checkpoints are written to the
working directory. Rendering (`grapple_render/`) needs a headless OpenGL backend:
`export MUJOCO_GL=egl` (GPU) or `osmesa` (CPU).

## Repository layout

| File | Role |
|---|---|
| `ffsm_core_se3.py` | Free-floating base + 6R arm: kinematics, mass matrix, momentum map, base reconstruction Φ, dynamics, tumbling target, collision clearances, `verify3d` certificate |
| `ffsm_demo_se3.py` | Scenario sampler (`rand_inst3d`) and momentum-aware multi-start IK demonstrator |
| `ffsm_dataset_se3.py` | Certified demonstration dataset (1,200 trajectories) |
| `ffsm_fm_se3.py` | Conditional flow matching on joint trajectories (the method) |
| `ffsm_ddpm_se3.py`, `ffsm_regress_se3.py` | Diffusion and Regression baselines (same backbone) |
| `ffsm_reach_eval_se3.py` | Shared reach metric, Cold / Oracle postures, checkpoint loader |
| `ffsm_baselines_eval_se3.py` | Table I and Table III |
| `ablate_projection.py` | Table I fixed-base execution ablation |
| `bootstrap_ci.py` | Bootstrap CIs and the paired FM − DDPM difference |
| `table2_sampling_cost.py` | Table II |
| `scp_warmstart_study_se3.py` | Reduced-coordinate sequential convex refinement; Table IV |
| `build_mjcf_se3.py`, `mujoco_crossval.py` | MuJoCo model and cross-validation (Sec. VII-F) |
| `mujoco_execute.py`, `mujoco_closedloop.py` | Table V (open-loop, PD tracking, MPC) |
| `ffsm_ood_eval_se3.py` | Table VI |
| `verify_prop2.py`, `test_se3.py`, `test_se3_gate.py` | Numerical checks of the propositions and the core |
| `grapple_render/` | MuJoCo rendering for Figs. 3–4 |

## System

50 kg base with I₀ = diag(8, 8, 10) kg·m², 6-DOF arm (10.1 kg, 1.57 m reach) mounted
0.7 m from the base origin, link masses [3, 3, 2, 1, 0.6, 0.5] kg, link lengths
[0.15, 0.60, 0.50, 0.12, 0.08, 0.12] m, torque limits [12, 12, 8, 4, 3, 3] N·m.
Scenarios: target center at 1.8–2.2 m, near-face grasp handle, ‖ω_g‖ ≤ 0.17 rad/s,
N = 60 nodes over t_f = 6 s. Feasibility: forward RK4 integration of the commanded
torques from rest, terminal position error < 5 cm, orientation error < 3°, no
collision, torque limits satisfied.

## Reproducing the paper

### 1. Certified dataset and training

```bash
python ffsm_dataset_se3.py --n_unique 1200 --N 60 --tf 6.0 --workers 16 --out ffsm_grapple_se3.pkl

python ffsm_fm_se3.py      --data ffsm_grapple_se3.pkl --epochs 3000 --out ffsm_fm_se3.pt
python ffsm_ddpm_se3.py    --data ffsm_grapple_se3.pkl --epochs 4000 --T 200 --out ffsm_ddpm_se3.pt
python ffsm_regress_se3.py --data ffsm_grapple_se3.pkl --epochs 4000 --out ffsm_regress_se3.pt
```

All three share the same pre-norm transformer (hidden 256, 6 layers, 8 heads, 4.96 M
parameters), AdamW (lr 1e-3, cosine schedule), batch 128. Nearest-neighbor needs no
training.

### 2. Results (held-out scenarios from disjoint seeds)

```bash
# Table I (+ Table III tail, recoil/cold-error correlation) -- 500 scenarios
python ffsm_baselines_eval_se3.py --n 500 --ckpt ffsm_fm_se3.pt --ddpm_ckpt ffsm_ddpm_se3.pt \
    --regress_ckpt ffsm_regress_se3.pt --data ffsm_grapple_se3.pkl
python bootstrap_ci.py reach_raw.npz                     # paired FM - DDPM CI (Sec. VII-C)
python ablate_projection.py --n 500 --ckpt ffsm_fm_se3.pt  # Table I fixed-base ablation

# Table II -- sampling cost and reach, 200 scenarios
python table2_sampling_cost.py --n 200 --ckpt ffsm_fm_se3.pt --ddpm_ckpt ffsm_ddpm_se3.pt

# Table IV -- solver-side warm-start study (same 500 scenarios as Table I)
python scp_warmstart_study_se3.py --n 500 --ckpt ffsm_fm_se3.pt \
    --ddpm_ckpt ffsm_ddpm_se3.pt --regress_ckpt ffsm_regress_se3.pt

# Sec. VII-F / Table V -- MuJoCo (viscous damping, +10% inertia, 2% torque noise), 200 scenarios
python mujoco_crossval.py
python mujoco_execute.py --n 200 --perturb --ckpt ffsm_fm_se3.pt \
    --ddpm_ckpt ffsm_ddpm_se3.pt --regress_ckpt ffsm_regress_se3.pt --data ffsm_grapple_se3.pkl
python mujoco_closedloop.py --mode pd  --n 200 --perturb --ckpt ffsm_fm_se3.pt
python mujoco_closedloop.py --mode mpc --n 200 --perturb --ckpt ffsm_fm_se3.pt

# Table VI -- out-of-distribution reach, 200 scenarios
python ffsm_ood_eval_se3.py --n 200 --ckpt ffsm_fm_se3.pt --data ffsm_grapple_se3.pkl

# Propositions
python verify_prop2.py
```

Solver-side timings come from an unoptimized finite-difference implementation and are
best read as ratios between initializations.

### 3. Figures 3–4 (MuJoCo rendering)

```bash
cd grapple_render
python build_render_mjcf.py --out ffsm_render.xml           # render model, inertias from FFSM3DParams
MUJOCO_GL=egl python run_selftest.py --model ffsm_render.xml  # frame conventions, Prop. 1 in MuJoCo
python check_model_match.py --model ffsm_render.xml --ckpt ../ffsm_fm_se3.pt
python export_trajectories.py --model ffsm_render.xml --ckpt ../ffsm_fm_se3.pt --out traj/
MUJOCO_GL=egl python run_render.py --npz 'traj/*.npz' --model ffsm_render.xml --out fig/ --no-video
python plot_fig4.py --npz 'traj/*.npz' --out fig/fig_error_vs_time.pdf
```

`export_trajectories.py` draws held-out scenarios (seed0 900000), samples the model once
per scenario and picks the median-recoil scenario in each of the bins (5–15°, 20–30°,
35–45°) without looking at reach error. The momentum-projected run is drawn as the solid
arm and the recoil-naive run as a translucent ghost in the same scene. Drop `--no-video`
to also write HUD video clips.

## Citation

Citation details will be added once the paper is published.

## License

MIT — see [LICENSE](LICENSE).
