"""bootstrap_ci.py -- bootstrap confidence intervals on reach_raw.npz (Sec. VII-C).

Median and 90th-percentile reach per method with 95% CIs, and the paired FM - DDPM
mean difference with its 95% CI (same scenarios, so the difference is bootstrapped).

    python3 bootstrap_ci.py reach_raw.npz
"""
import numpy as np, sys
d = np.load(sys.argv[1]); out = d['out']          # (n_scenarios, n_columns)
NAMES = ['recoil','cold','NN','regress','DDPM','FM','oracle']

def ci(x, stat=np.median, B=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    s = np.array([stat(rng.choice(x, x.size, replace=True)) for _ in range(B)])
    return stat(x), np.percentile(s, 2.5), np.percentile(s, 97.5)

print(f"{'method':>9} {'median [95% CI]':>26} {'p90 [95% CI]':>26}")
for k, nm in enumerate(NAMES[1:], start=1):
    m, lo, hi = ci(out[:, k])
    p, plo, phi = ci(out[:, k], stat=lambda v: np.percentile(v, 90))
    print(f"{nm:>9}   {m:5.2f} [{lo:5.2f},{hi:5.2f}]   {p:6.2f} [{plo:6.2f},{phi:6.2f}]")

# paired FM vs DDPM -- same scenarios, so bootstrap the DIFFERENCE
diff = out[:, NAMES.index('FM')] - out[:, NAMES.index('DDPM')]
m, lo, hi = ci(diff, stat=np.mean)
print(f"\npaired FM-DDPM mean difference: {m:+.2f} cm [{lo:+.2f}, {hi:+.2f}]")
print("interval spanning zero => the two are statistically indistinguishable")
