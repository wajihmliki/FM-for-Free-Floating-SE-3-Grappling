"""SE(3) gate self-test: inverse dynamics (momentum-consistent) + verify3d on a matched target."""
import numpy as np, ffsm_core_se3 as fc3
p=fc3.FFSM3DParams(); N=30; tf=6.0; dt=tf/N; s=np.linspace(0,1,N+1)
n=p.n; th0=np.linspace(0.3,0.4,n); thf=np.linspace(0.8,-0.4,n)
ease=10*s**3-15*s**4+6*s**5; de=(30*s**2-60*s**3+30*s**4)/tf; dde=(60*s-180*s**2+120*s**3)/tf**2
th=th0+np.outer(ease,thf-th0); thd=np.outer(de,thf-th0); thdd=np.outer(dde,thf-th0)
base0=np.concatenate([np.zeros(3), np.array([1.,0,0,0])])
Q,U,tau,bres=fc3.inverse_dynamics3d(th,thd,thdd,base0,dt,p)
print(f"[id]   base-force residual {bres:.1e} (want ~0)   max|tau| {np.abs(tau).max():.2f}")
kend=fc3.fk3d(Q[-1],p); ee_q=fc3.ee_quat(Q[-1],p)
c_center=kend['ee_pos'] - fc3.rotmat(ee_q)@np.array([-0.45,0,0])
tgt=fc3.Target3D(p0=c_center, v=np.zeros(3), quat0=ee_q, omega=np.zeros(3),
                 grasp_offset=np.array([-0.45,0,0]), grasp_quat=np.array([1.,0,0,0]))
inst=fc3.build_inst3d(p,tgt,th0,base0,tf,N)
info=fc3.verify3d(inst, tau, substeps=10)
print(f"[gate] feasible={info['feasible']} pos={info['pos_err']*100:.2f}cm ang={info['ang_err_deg']:.2f}deg "
      f"vio={info['max_vio']:.3f} |dP|={info['P_drift']:.1e} |dL|={info['L_drift']:.1e}")
