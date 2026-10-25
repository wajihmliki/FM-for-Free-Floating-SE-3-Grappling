"""ffsm_regress_se3.py -- deterministic Regression baseline.

Same backbone, conditioning, data and certificate as the flow model, but trained to
directly predict the joint trajectory theta(t) from the conditioning with a
mean-squared-error loss (no generative process).

    python3 ffsm_regress_se3.py --smoke
    python3 ffsm_regress_se3.py --data ffsm_grapple_se3.pkl --epochs 4000 --out ffsm_regress_se3.pt
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ffsm_fm_se3 import FFSM3DVelocityNet, load_se3   # reuse the exact backbone + loader


def train_regress(net, theta, cond, n_epochs=4000, batch=128, lr=1e-3, device='cpu', log_every=400):
    theta = theta.to(device); cond = cond.to(device); N, T, D = theta.shape
    m = theta.mean((0,1), keepdim=True); s = theta.std((0,1), keepdim=True).clamp_min(1e-6)
    cm = cond.mean(0, keepdim=True); cs = cond.std(0, keepdim=True).clamp_min(1e-6)
    y = (theta-m)/s; ca = (cond-cm)/cs
    net = net.to(device); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    zero_t = torch.zeros(batch, device=device)
    for ep in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch]; c = ca[idx]; tgt = y[idx]; B = c.size(0)
        x0 = torch.zeros(B, T, D, device=device)                 # fixed zero input; cond drives the map
        pred = net(x0, zero_t[:B], c)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        if ep % log_every == 0 or ep == n_epochs-1: print(f"  [SE3-REGRESS] ep {ep:4d} loss={loss.item():.5f}")
    return net, dict(mean=m.cpu(), std=s.cpu(), cond_mean=cm.cpu(), cond_std=cs.cpu())


@torch.no_grad()
def sample_regress(net, cond, stats, seq_len, device='cpu', th0_slice=None):
    """Deterministic forward -> theta. CVF-style start anchor (node 0 = th0) for a fair comparison."""
    net.eval(); cond = cond.to(device); B = cond.size(0)
    m = stats['mean'].to(device); s = stats['std'].to(device)
    cm = stats['cond_mean'].to(device); cs = stats['cond_std'].to(device)
    cn = (cond-cm)/cs
    x0 = torch.zeros(B, seq_len, m.shape[-1], device=device)
    pred = net(x0, torch.zeros(B, device=device), cn)
    if th0_slice is not None:
        pred[:, 0, :] = (cond[:, th0_slice[0]:th0_slice[1]] - m[0,0]) / s[0,0]
    return pred*s + m


def _smoke():
    torch.manual_seed(0)
    T=20; th=torch.randn(48,T,6)*0.3; cd=torch.randn(48,24)
    net=FFSM3DVelocityNet(seq_len=T,cond_dim=24,dim=6,hidden=96,n_layers=3,n_heads=4)
    net,st=train_regress(net,th,cd,n_epochs=400,batch=24,log_every=200)
    s=sample_regress(net,cd[:8],st,seq_len=T,th0_slice=(0,6)).cpu().numpy()
    print("shape", s.shape, "-> SMOKE OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ffsm_grapple_se3.pkl'); ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--n_layers', type=int, default=6); ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='ffsm_regress_se3.pt'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke: _smoke(); raise SystemExit
    theta, cond, meta = load_se3(a.data); T = theta.shape[1]; n = theta.shape[2]
    print(f"SE3-REGRESS: theta {tuple(theta.shape)} cond_dim {cond.shape[1]} T={T} n={n} device {a.device}")
    net = FFSM3DVelocityNet(seq_len=T, cond_dim=cond.shape[1], dim=n, hidden=a.hidden, n_layers=a.n_layers, n_heads=a.n_heads)
    net, stats = train_regress(net, theta, cond, n_epochs=a.epochs, batch=a.batch, device=a.device, log_every=max(a.epochs//10,1))
    torch.save({'state_dict':net.state_dict(),'stats':stats,'seq_len':T,'cond_dim':cond.shape[1],'dim':n,
                'hidden':a.hidden,'n_layers':a.n_layers,'n_heads':a.n_heads,'meta':meta}, a.out)
    print(f"saved {a.out}")
