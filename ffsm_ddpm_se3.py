"""ffsm_ddpm_se3.py -- Diffusion baseline (DDPM) for the SE(3) grapple warm start.

Same backbone, standardized conditioning, joint-trajectory target, momentum
reconstruction and certificate as ffsm_fm_se3.py; only the generative process differs:
DDPM with a cosine noise schedule (T=200) and epsilon-prediction, sampled with the full
ancestral chain.

    python3 ffsm_ddpm_se3.py --smoke
    python3 ffsm_ddpm_se3.py --data ffsm_grapple_se3.pkl --epochs 4000 --T 200 --out ffsm_ddpm_se3.pt
"""
import argparse, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ffsm_fm_se3 import FFSM3DVelocityNet, load_se3   # reuse the exact backbone + loader


def cosine_abar(T, s=0.008):
    t = torch.linspace(0, T, T+1)/T
    f = torch.cos((t+s)/(1+s)*np.pi/2)**2
    ab = f/f[0]; return ab.clamp(1e-5, 0.9999)          # abar[0..T]


def train_ddpm(net, theta, cond, T=200, n_epochs=4000, batch=128, lr=1e-3, device='cpu', log_every=400):
    theta = theta.to(device); cond = cond.to(device); N, Tlen, D = theta.shape
    m = theta.mean((0,1), keepdim=True); s = theta.std((0,1), keepdim=True).clamp_min(1e-6)
    cm = cond.mean(0, keepdim=True); cs = cond.std(0, keepdim=True).clamp_min(1e-6)
    x0a = (theta-m)/s; ca = (cond-cm)/cs; abar = cosine_abar(T).to(device)
    net = net.to(device); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch]; x0 = x0a[idx]; c = ca[idx]; B = x0.size(0)
        tt = torch.randint(1, T+1, (B,), device=device); ab = abar[tt].view(B,1,1)
        eps = torch.randn_like(x0); xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
        eps_pred = net(xt, tt.float()/T, c)
        loss = F.mse_loss(eps_pred, eps)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        if ep % log_every == 0 or ep == n_epochs-1: print(f"  [SE3-DDPM] ep {ep:4d} loss={loss.item():.5f}")
    return net, dict(mean=m.cpu(), std=s.cpu(), cond_mean=cm.cpu(), cond_std=cs.cpu(), T=T)


@torch.no_grad()
def sample_theta_ddpm(net, cond, stats, seq_len, n_steps=None, device='cpu', th0_slice=None, ancestral=True):
    """Sampling with CVF start-anchor -- same interface as FM's sample_theta.
       ancestral=True: full DDPM ancestral sampling over ALL T steps (fair, strongest baseline).
       ancestral=False: deterministic DDIM with n_steps (fast)."""
    net.eval(); cond = cond.to(device); B = cond.size(0); T = stats['T']
    m = stats['mean'].to(device); s = stats['std'].to(device)
    cm = stats['cond_mean'].to(device); cs = stats['cond_std'].to(device)
    cn = (cond-cm)/cs; abar = cosine_abar(T).to(device)
    x = torch.randn(B, seq_len, m.shape[-1], device=device)
    th0_std = ((cond[:, th0_slice[0]:th0_slice[1]]-m[0,0])/s[0,0]) if th0_slice else None
    if ancestral:
        for t in range(T, 0, -1):
            ab = abar[t]; ab_prev = abar[t-1]
            eps = net(x, torch.full((B,), float(t)/T, device=device), cn)
            x0 = (x - (1-ab).sqrt()*eps)/ab.sqrt()
            if th0_std is not None: x0[:, 0, :] = th0_std
            beta = 1 - ab/ab_prev
            mean = ab_prev.sqrt()*beta/(1-ab)*x0 + (ab/ab_prev).sqrt()*(1-ab_prev)/(1-ab)*x
            x = mean + (beta.sqrt()*torch.randn_like(x) if t > 1 else 0.0)
        return x*s + m
    steps = torch.linspace(T, 1, n_steps or 40).long().to(device)
    for i in range(len(steps)):
        t = steps[i]; ab = abar[t]
        eps = net(x, torch.full((B,), float(t)/T, device=device), cn)
        x0 = (x - (1-ab).sqrt()*eps)/ab.sqrt()
        if th0_std is not None: x0[:, 0, :] = th0_std
        if i < len(steps)-1:
            ab_prev = abar[steps[i+1]]; x = ab_prev.sqrt()*x0 + (1-ab_prev).sqrt()*eps
        else:
            x = x0
    return x*s + m


def _smoke():
    torch.manual_seed(0); np.random.seed(0)
    import ffsm_core_se3 as fc3
    p = fc3.FFSM3DParams(); Tn = 20; tf = 6.0; s = np.linspace(0,1,Tn)
    ease = 10*s**3-15*s**4+6*s**5; n = 48; ths=[]; cds=[]
    for k in range(n):
        th0 = np.array([0.2,0.5,0.4,0,0.3,0.])+np.random.randn(6)*0.05; thf = th0+np.random.randn(6)*0.5
        ths.append((th0+np.outer(ease,thf-th0)).astype(np.float32)); cds.append(np.concatenate([th0,np.zeros(18)]).astype(np.float32))
    theta = torch.tensor(np.stack(ths)); cond = torch.tensor(np.stack(cds))
    net = FFSM3DVelocityNet(seq_len=Tn, cond_dim=24, dim=6, hidden=96, n_layers=3, n_heads=4)
    net, stats = train_ddpm(net, theta, cond, T=100, n_epochs=500, batch=24, log_every=250)
    samp = sample_theta_ddpm(net, cond[:8], stats, seq_len=Tn, n_steps=30, th0_slice=(0,6)).cpu().numpy()
    print(f"[start-anchor] max|theta(0)-th0| = {np.abs(samp[:,0,:]-cond[:8,:6].numpy()).max():.3f}")
    print("SMOKE OK (DDPM trains + samples with start anchor)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ffsm_grapple_se3.pkl'); ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--T', type=int, default=200); ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--n_heads', type=int, default=8); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='ffsm_ddpm_se3.pt'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke: _smoke(); raise SystemExit
    theta, cond, meta = load_se3(a.data); Tlen = theta.shape[1]; n = theta.shape[2]
    print(f"SE3-DDPM: theta {tuple(theta.shape)} cond_dim {cond.shape[1]} T={Tlen} n={n} device {a.device}")
    net = FFSM3DVelocityNet(seq_len=Tlen, cond_dim=cond.shape[1], dim=n, hidden=a.hidden, n_layers=a.n_layers, n_heads=a.n_heads)
    net, stats = train_ddpm(net, theta, cond, T=a.T, n_epochs=a.epochs, batch=a.batch, device=a.device, log_every=max(a.epochs//10,1))
    torch.save({'state_dict':net.state_dict(),'stats':stats,'seq_len':Tlen,'cond_dim':cond.shape[1],'dim':n,
                'hidden':a.hidden,'n_layers':a.n_layers,'n_heads':a.n_heads,'meta':meta}, a.out)
    print(f"saved {a.out}")
