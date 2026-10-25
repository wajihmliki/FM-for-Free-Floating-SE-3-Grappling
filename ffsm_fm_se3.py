"""ffsm_fm_se3.py -- conditional flow matching on joint trajectories + momentum-consistent
base reconstruction (the method, Sec. IV-A/B).

The model generates the joint trajectory theta_{0:N} (n=6), conditioned on the 24-dim
grapple scenario c = (theta_0, p_g^0, R_g^0, omega_g, o_g, R_g, t_f). Training uses the
linear conditional path x_s = (1-s) x_0 + s x_1 and the velocity-regression objective;
sampling integrates the learned field with Euler steps and anchors node 0 to theta_0.
Joint trajectories and conditions are standardized over the training set.

The spacecraft base is NOT generated: it is reconstructed from the generated joints via
the zero-momentum map (ffsm_core_se3.inverse_dynamics3d / rollout_reduced_se3_fast), so
momentum conservation holds by construction (Prop. 1).

    python3 ffsm_fm_se3.py --data ffsm_grapple_se3.pkl --epochs 3000 --out ffsm_fm_se3.pt
"""
import math, argparse, pickle
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import ffsm_core_se3 as fc3


# ---------------------------- backbone (mirrors JointVelocityNetV2) ----------------------------
class SinusoidalT(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        fr = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device).float() / max(half-1, 1))
        a = t.unsqueeze(-1) * fr.unsqueeze(0); return torch.cat([a.sin(), a.cos()], dim=-1)

class FFSM3DVelocityNet(nn.Module):
    def __init__(self, seq_len, cond_dim, dim, hidden=256, n_layers=6, n_heads=8, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        self.t_embed = nn.Sequential(SinusoidalT(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.out_norm = nn.LayerNorm(hidden); self.out_proj = nn.Linear(hidden, dim)
    def forward(self, x, t, cond):
        h = self.in_proj(x) + self.pos_emb[:, :x.size(1), :]
        h = h + (self.t_embed(t) + self.cond_embed(cond))[:, None, :]
        h = self.transformer(h); h = self.out_norm(h); return self.out_proj(h)


# ---------------------------- training (Euclidean FM on joints; normalized cond) ----------------------------
def train_ffsm3d_fm(net, theta, cond, n_epochs=3000, batch=128, lr=1e-3, device='cpu', log_every=300):
    theta = theta.to(device); cond = cond.to(device); N, T, D = theta.shape
    m = theta.mean((0,1), keepdim=True); s = theta.std((0,1), keepdim=True).clamp_min(1e-6)
    cm = cond.mean(0, keepdim=True); cs = cond.std(0, keepdim=True).clamp_min(1e-6)
    x1a = (theta-m)/s; ca = (cond-cm)/cs
    net = net.to(device); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch]; x1 = x1a[idx]; c = ca[idx]; B = x1.size(0)
        x0 = torch.randn_like(x1); t = torch.rand(B, device=device)
        xt = (1-t.view(B,1,1))*x0 + t.view(B,1,1)*x1
        loss = F.mse_loss(net(xt, t, c), x1-x0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if ep % log_every == 0 or ep == n_epochs-1: print(f"  [SE3-FM] ep {ep:4d} loss={loss.item():.5f}")
    return net, dict(mean=m.cpu(), std=s.cpu(), cond_mean=cm.cpu(), cond_std=cs.cpu())


@torch.no_grad()
def sample_theta(net, cond, stats, seq_len, n_steps=50, device='cpu', th0_slice=None):
    net.eval(); cond = cond.to(device); B = cond.size(0)
    m = stats['mean'].to(device); s = stats['std'].to(device)
    cm = stats['cond_mean'].to(device); cs = stats['cond_std'].to(device)
    cn = (cond-cm)/cs; x = torch.randn(B, seq_len, m.shape[-1], device=device)
    th0_std = ((cond[:, th0_slice[0]:th0_slice[1]] - m[0,0]) / s[0,0]) if th0_slice else None
    dt = 1.0/n_steps
    for i in range(n_steps):
        tv = i*dt; v = net(x, torch.full((B,), tv, device=device), cn).clone()
        if th0_std is not None: v[:, 0, :] = (th0_std - x[:, 0, :]) / max(1.0-tv, 1e-3)
        x = x + dt*v
    return x*s + m


# ---------------------------- momentum-null-space projection on SE(3) ----------------------------
def _smooth_joints(th, dt, lam=2e-3):
    from scipy.interpolate import make_smoothing_spline
    T = th.shape[0]; t = np.arange(T)*dt; out = [np.zeros_like(th) for _ in range(3)]
    for j in range(th.shape[1]):
        sp = make_smoothing_spline(t, th[:, j], lam=lam)
        out[0][:, j] = sp(t); out[1][:, j] = sp(t, 1); out[2][:, j] = sp(t, 2)
    return out

def fm_warmstart3d(theta, inst, project=True, smooth=2e-3):
    """theta (T,n) generated joints -> SE(3) warm-start (Q, tau).
       project=True  -> base pose = momentum-consistent reconstruction Phi (ours).
       project=False -> base orientation integrated as a Euclidean R^4 quaternion
                        (off SO(3)), for comparison only."""
    p, N, dt, base0 = inst['p'], inst['N'], inst['dt'], inst['base0']
    th, thd, thdd = _smooth_joints(np.asarray(theta, float), dt, smooth)
    if project:
        Q, U, tau, _ = fc3.inverse_dynamics3d(th, thd, thdd, base0, dt, p)
        return Q, tau
    # FM-Eucl: base reconstructed by EUCLIDEAN (R^4) quaternion integration (off SO(3)); real torque
    Qe, Ue = fc3.rollout_reduced_se3_eucl(base0, th, thd, dt, p, substeps=3)
    tau = fc3.torque_from_path(Qe, Ue, dt, p)
    return Qe, tau


def load_se3(path):
    d = pickle.load(open(path, 'rb'))
    return (torch.from_numpy(d['theta']).float(), torch.from_numpy(d['cond']).float(), d['meta'])


def _smoke():
    torch.manual_seed(0); np.random.seed(0)
    p = fc3.FFSM3DParams(); T = 20; tf = 6.0; dt = tf/(T-1); n = 48
    s = np.linspace(0,1,T); ease = 10*s**3-15*s**4+6*s**5; de = (30*s**2-60*s**3+30*s**4)/tf
    base0 = np.concatenate([np.zeros(3), [1.,0,0,0]]); ths=[]; cds=[]
    for k in range(n):
        th0 = np.array([0.2,0.5,0.4,0,0.3,0.]) + np.random.randn(6)*0.05
        thf = th0 + np.random.randn(6)*0.5
        th = th0 + np.outer(ease, thf-th0); ths.append(th.astype(np.float32))
        cds.append(np.concatenate([th0, np.zeros(24-6)]).astype(np.float32))
    theta = torch.tensor(np.stack(ths)); cond = torch.tensor(np.stack(cds))
    print(f"smoke: theta {tuple(theta.shape)} cond {tuple(cond.shape)}")
    net = FFSM3DVelocityNet(seq_len=T, cond_dim=24, dim=6, hidden=96, n_layers=3, n_heads=4)
    net, stats = train_ffsm3d_fm(net, theta, cond, n_epochs=500, batch=24, lr=1e-3, log_every=250)
    samp = sample_theta(net, cond[:8], stats, seq_len=T, n_steps=50, th0_slice=(0,6)).cpu().numpy()
    start_err = np.abs(samp[:,0,:] - cond[:8,:6].numpy()).max()
    print(f"[start-anchor] max|theta(0)-th0| = {start_err:.3f}")
    # momentum projection: FM-Riem base is SO(3)-consistent (base_resid ~0)
    tgt = fc3.Target3D()
    inst = fc3.build_inst3d(p, tgt, th0=samp[0,0], base0=base0, tf=tf, N=T-1)
    Qr, tr = fm_warmstart3d(samp[0], inst, project=True)
    th = Qr[:,7:]; thd=np.gradient(th,dt,axis=0); thdd=np.gradient(thd,dt,axis=0)
    _,_,_,bres = fc3.inverse_dynamics3d(th, thd, thdd, base0, dt, p)
    # base orientation is a valid unit quaternion trajectory on SO(3)
    qn = np.linalg.norm(Qr[:,3:7], axis=1)
    print(f"[reconstruction] base-force residual {bres:.1e} (momentum-consistent) | quat-norm dev {np.abs(qn-1).max():.1e}")
    print("SMOKE OK" if start_err<0.15 and bres<1e-6 else "check")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ffsm_grapple_se3.pkl'); ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--n_layers', type=int, default=6); ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='ffsm_fm_se3.pt'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke: _smoke(); raise SystemExit
    theta, cond, meta = load_se3(a.data); T = theta.shape[1]; n = theta.shape[2]
    print(f"SE3-FM: theta {tuple(theta.shape)} cond_dim {cond.shape[1]} T={T} n={n} device {a.device}")
    net = FFSM3DVelocityNet(seq_len=T, cond_dim=cond.shape[1], dim=n, hidden=a.hidden, n_layers=a.n_layers, n_heads=a.n_heads)
    net, stats = train_ffsm3d_fm(net, theta, cond, n_epochs=a.epochs, batch=a.batch, device=a.device,
                                 log_every=max(a.epochs//10,1))
    torch.save({'state_dict':net.state_dict(),'stats':stats,'seq_len':T,'cond_dim':cond.shape[1],'dim':n,
                'hidden':a.hidden,'n_layers':a.n_layers,'n_heads':a.n_heads,'meta':meta}, a.out)
    print(f"saved {a.out}")
