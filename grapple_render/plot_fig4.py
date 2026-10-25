"""plot_fig4.py -- Fig. 4: gripper-to-grasp distance over time for the Fig. 3 scenarios.

Plots the position error between the gripper and the grasp frame over the whole
maneuver from the exported .npz arrays, and checks that the terminal values reproduce
meta['reach_fm_cm'] / meta['reach_cold_cm'] (the Fig. 3 insets) to 0.1 cm before
writing the figure.

    python3 plot_fig4.py --npz 'traj/*.npz' --out fig/fig_error_vs_time.pdf
"""
import argparse, glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FM_C, CD_C = "#18b26b", "#e2503c"
STYLES = ["-", "--", ":"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="traj/*.npz")
    ap.add_argument("--out", default="fig/fig_error_vs_time.pdf")
    ap.add_argument("--tol", type=float, default=0.1, help="cm agreement with insets")
    ap.add_argument("--force", action="store_true", help="write even if verification fails")
    a = ap.parse_args()

    files = sorted(glob.glob(a.npz))
    if not files:
        sys.exit(f"no files matched {a.npz}")

    rows, bad = [], False
    for f in files:
        d = np.load(f, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        t = np.asarray(d["t"], float)
        grasp = np.asarray(d["grasp"], float)[:, :3]
        e_fm = np.linalg.norm(np.asarray(d["grip_fm"], float)[:, :3] - grasp, axis=1) * 100
        e_cd = np.linalg.norm(np.asarray(d["grip_cold"], float)[:, :3] - grasp, axis=1) * 100
        ok_fm = abs(e_fm[-1] - meta["reach_fm_cm"]) <= a.tol
        ok_cd = abs(e_cd[-1] - meta["reach_cold_cm"]) <= a.tol
        flag = "" if (ok_fm and ok_cd) else "   <-- MISMATCH"
        if not (ok_fm and ok_cd):
            bad = True
        print(f"  rho={meta['recoil_deg']:5.1f} deg   "
              f"FM curve {e_fm[-1]:6.2f} vs inset {meta['reach_fm_cm']:6.2f}   "
              f"COLD curve {e_cd[-1]:6.2f} vs inset {meta['reach_cold_cm']:6.2f}{flag}")
        rows.append((meta["recoil_deg"], t, e_fm, e_cd))

    if bad and not a.force:
        sys.exit("\nterminal values do not match the Fig. 3 insets -- not writing the figure "
                 "(use --force to override).")

    rows.sort(key=lambda r: r[0])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (rho, t, e_fm, e_cd) in enumerate(rows):
        ls = STYLES[i % len(STYLES)]
        ax.semilogy(t, e_fm, ls, color=FM_C, lw=1.6)
        ax.semilogy(t, e_cd, ls, color=CD_C, lw=1.6)
    ax.axhline(5.0, color="0.55", lw=0.8, ls=(0, (4, 3)))
    ax.text(t[-1], 5.4, "5 cm gate", ha="right", va="bottom", fontsize=6.5, color="0.4")
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("gripper-to-grasp (cm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", lw=0.3, alpha=0.35)
    h = [plt.Line2D([], [], color=FM_C, lw=1.6, label="momentum-projected"),
         plt.Line2D([], [], color=CD_C, lw=1.6, label="recoil-naive")]
    h += [plt.Line2D([], [], color="0.35", lw=1.4, ls=STYLES[i % len(STYLES)],
                     label=fr"$\rho={rows[i][0]:.0f}^\circ$") for i in range(len(rows))]
    ax.legend(handles=h, fontsize=6.2, loc="lower left", framealpha=0.9, ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
