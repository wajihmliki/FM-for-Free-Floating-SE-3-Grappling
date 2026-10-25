"""table2_sampling_cost.py -- Table II: sampling cost and reach for 200 scenarios.

FM with 10 / 25 / 50 Euler steps vs DDPM with the full ancestral chain (T=200). Timing
is warmed up and CUDA-synchronized; ms/NFE is printed as a consistency check (every
configuration evaluates the same backbone once per step).

    python3 table2_sampling_cost.py --ckpt ffsm_fm_se3.pt --ddpm_ckpt ffsm_ddpm_se3.pt --n 200
"""
import argparse, time
import numpy as np, torch
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from ffsm_dataset_se3 import cond_vec
from ffsm_reach_eval_se3 import _mj, reach_of, load as load_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ffsm_fm_se3.pt")
    ap.add_argument("--ddpm_ckpt", default="ffsm_ddpm_se3.pt")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--tf", type=float, default=6.0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed0", type=int, default=400000)
    ap.add_argument("--reps", type=int, default=30); ap.add_argument("--warm", type=int, default=15)
    a = ap.parse_args()

    from ffsm_fm_se3 import sample_theta
    from ffsm_ddpm_se3 import sample_theta_ddpm
    net, ck = load_gen(a.ckpt, a.device)
    dnet, dck = load_gen(a.ddpm_ckpt, a.device)
    N = ck["seq_len"] - 1; p = fc3.FFSM3DParams(); ease, de = _mj(N, a.tf)
    insts = [d3.rand_inst3d(a.seed0 + i, N, a.tf, p) for i in range(a.n)]
    cond = torch.from_numpy(np.stack([cond_vec(i) for i in insts])).float().to(a.device)

    def bench(fn):
        for _ in range(a.warm): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / a.n * 1000)
        return float(np.median(ts))

    def reach_of_samples(th):
        return float(np.median([reach_of(insts[j], th[j][-1], ease, de)[0] for j in range(a.n)]))

    cfgs = []
    for k in (10, 25, 50):
        cfgs.append((f"FM, {k} steps", k,
                     lambda k=k: sample_theta(net, cond, ck["stats"], seq_len=ck["seq_len"],
                                              device=a.device, th0_slice=(0, ck["dim"]), n_steps=k)))
    cfgs.append(("DDPM, ancestral (T=200)", 200,
                 lambda: sample_theta_ddpm(dnet, cond, dck["stats"], seq_len=dck["seq_len"],
                                           device=a.device, th0_slice=(0, dck["dim"]),
                                           ancestral=True)))

    print(f"\n=== Table II: sampling cost and reach (n={a.n}, {a.reps} timed reps "
          f"after {a.warm} warmup) ===")
    print(f"{'Method':>26} {'NFE':>5} {'ms/sample':>10} {'Reach (cm)':>11} {'ms/NFE':>8}")
    rows = []
    for name, nfe, fn in cfgs:
        ms = bench(fn)
        reach = reach_of_samples(fn().cpu().numpy())
        rows.append((name, nfe, ms, reach))
        print(f"{name:>26} {nfe:>5} {ms:>10.2f} {reach:>11.1f} {ms/nfe:>8.4f}")

    per = np.array([ms / nfe for _, nfe, ms, _ in rows])
    spread = per.max() / per.min()
    print(f"\nms/NFE spread across rows: {spread:.2f}x", end="  ")
    print("(consistent: one backbone call per step)" if spread < 1.25
          else "(inconsistent timing across rows)")
    fm10 = next(r for r in rows if r[1] == 10)
    anc = next(r for r in rows if r[0].startswith("DDPM, ancestral"))
    print(f"FM(10) vs DDPM ancestral: {anc[2]/fm10[2]:.1f}x fewer ms, "
          f"{anc[1]/fm10[1]:.0f}x fewer NFE, reach {fm10[3]:.1f} vs {anc[3]:.1f} cm")


if __name__ == "__main__":
    main()
