"""
figure.py -- assembles the paper figure.

Layout: one row per scenario (low / mid / high recoil), four columns across
the maneuver. Panels are rasterised MuJoCo renders; ALL TEXT IS VECTOR, so it
stays sharp in the PDF. Saves PDF (for
\\includegraphics) and PNG (for slides).

IEEE conference column width is 3.5 in; double-column is 7.16 in. Default
here is double-column, which is what a 3x4 filmstrip needs.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "pdf.fonttype": 42,          # TrueType, not Type3 -- IEEE PDF checks want this
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
})

C_FM = "#22d97a"
C_COLD = "#f2593f"
C_GRASP = "#ffd633"


def filmstrip(panel_paths, metas, out_pdf, out_png=None,
              col_labels=None, width_in=7.16, crop=None, dpi=400, insets=None):
    """
    panel_paths : list of rows, each a list of image paths (same length)
    metas       : one dict per row (needs recoil_deg, reach_fm_cm, reach_cold_cm)
    crop        : optional (left, top, right, bottom) fractions to trim dead space
    """
    nrow, ncol = len(panel_paths), len(panel_paths[0])
    im0 = Image.open(panel_paths[0][0])
    if crop:
        l, t, r, b = crop
        W, H = im0.size
        box = (int(l * W), int(t * H), int(r * W), int(b * H))
        im0 = im0.crop(box)
    else:
        box = None
    ar = im0.size[1] / im0.size[0]

    panel_w = width_in / ncol
    fig_h = panel_w * ar * nrow + 0.62
    fig = plt.figure(figsize=(width_in, fig_h), dpi=dpi)

    gs = fig.add_gridspec(nrow, ncol, left=0.052, right=0.998,
                          top=0.928, bottom=0.052, wspace=0.012, hspace=0.02)

    for i in range(nrow):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            im = Image.open(panel_paths[i][j])
            if box:
                im = im.crop(box)
            ax.imshow(np.asarray(im), interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#3a4152"); sp.set_linewidth(0.5)
            if i == 0 and col_labels:
                ax.set_title(col_labels[j], fontsize=7.2, pad=3, color="#111111")
            if j == 0:
                m = metas[i]
                ax.set_ylabel(f"$\\rho$ = {m['recoil_deg']:.0f}$^\\circ$",
                              fontsize=7.4, labelpad=3)
            if j == ncol - 1 and insets and insets[i]:
                iim = Image.open(insets[i])
                iw, ih = iim.size
                side = 0.40
                axi = ax.inset_axes([1 - side - 0.016, 1 - side / ar - 0.022,
                                     side, side / ar])
                axi.imshow(np.asarray(iim), interpolation="lanczos")
                axi.set_xticks([]); axi.set_yticks([])
                for sp in axi.spines.values():
                    sp.set_color(C_GRASP); sp.set_linewidth(0.9)
            if j == ncol - 1:
                m = metas[i]
                ax.text(0.975, 0.055,
                        f"{m['reach_fm_cm']:.1f} cm  vs  {m['reach_cold_cm']:.1f} cm",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.6, color="white",
                        bbox=dict(boxstyle="round,pad=0.24", fc="#0b0e14cc", ec="none"))

    handles = [
        Line2D([], [], color=C_FM, lw=2.4, label="momentum-projected gripper path"),
        Line2D([], [], color=C_COLD, lw=2.4, label="recoil-naive gripper path (ghost arm)"),
        Line2D([], [], color=C_GRASP, lw=2.4, label="moving grasp frame $g(t)$ path"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, 1.002), handlelength=1.6,
               columnspacing=1.9)

    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.012)
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    return out_pdf


def error_curves(trajs, out_pdf, out_png=None, width_in=3.5):
    """Companion plot: gripper-to-grasp distance vs time, all scenarios.
    Single column. Shows the divergence, not just the endpoint."""
    fig, ax = plt.subplots(figsize=(width_in, 2.35), dpi=400)
    styles = ["-", "--", ":", "-."]
    rho_handles = []
    for i, (t, e_fm, e_cd, m) in enumerate(trajs):
        ls = styles[i % len(styles)]
        ax.plot(t, np.array(e_cd) * 100, color=C_COLD, lw=1.3, ls=ls)
        ax.plot(t, np.array(e_fm) * 100, color=C_FM, lw=1.3, ls=ls)
        rho_handles.append(Line2D([], [], color="0.35", lw=1.2, ls=ls,
                                  label=f"$\\rho$ = {m['recoil_deg']:.0f}$^\\circ$"))
    ax.set_xlabel("time (s)", fontsize=7.5)
    ax.set_ylabel("gripper-to-grasp (cm)", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-2)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    leg1 = ax.legend(handles=[Line2D([], [], color=C_FM, lw=1.4, label="momentum-projected"),
                              Line2D([], [], color=C_COLD, lw=1.4, label="recoil-naive")],
                     fontsize=6.6, frameon=False, loc="lower left")
    ax.add_artist(leg1)
    # linestyle encodes the scenario
    ax.legend(handles=rho_handles, fontsize=6.6, frameon=False,
              loc="lower right", ncol=1, columnspacing=1.0, handlelength=1.8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    fig.tight_layout(pad=0.25)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.012)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)
    return out_pdf
