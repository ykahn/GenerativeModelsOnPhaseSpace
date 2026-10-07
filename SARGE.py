"""Generates events via the SARGE algorithm

Usage:
    python SARGE.py --n-events 100000 --n-particles 10 --xi 20.0
"""


import torch
import math

import argparse
import os
import sys

def get_b_x_from_qs(qs, energy=1.0):
    """Return CM boost vector b and conformal parameter x for given q vectors."""
    Qs = qs.sum(dim=1)                                           # (nevents, 3)
    Q0s = torch.sum(torch.linalg.norm(qs, dim=2), dim=1)        # (nevents,)

    Ms2 = Q0s ** 2 - torch.linalg.norm(Qs, dim=1) ** 2
    Ms = torch.sqrt(Ms2)

    bs = -Qs / Ms[:, None]
    xs = energy / Ms
    return bs, xs


def Hmu(inputvecs, bs):
    """Lorentz boost map (conformal rescaling omitted -- applied externally)."""
    gammas = torch.sqrt(1 + torch.linalg.norm(bs, axis=1) ** 2)
    As = 1 / (1 + gammas)
    bdotvecs = torch.einsum("ac,abc->ab", bs, inputvecs)
    outputvecs = (
        inputvecs
        + bs[:, None, :] * torch.linalg.norm(inputvecs, dim=2)[:, :, None]
        + As[:, None, None] * bdotvecs[:, :, None] * bs[:, None, :]
    )
    return outputvecs


def qs_to_ps(qs, energy=1.0):
    """q-space to p-space conformal map."""
    bs, xs = get_b_x_from_qs(qs, energy=energy)
    return Hmu(qs, bs) * xs[:, None, None]

def myboost_massless(p,n,y,sign=+1):
    gamma = torch.cosh(y)
    gammabeta = -torch.sinh(y)*sign
    pdotn = (p * n).sum(-1)
    E = torch.linalg.norm(p,axis=-1)
    Enew = gamma*E - gammabeta*pdotn
    pnew = -gammabeta[...,None]*n*E[...,None] + p + gammabeta[...,None]**2/(1+gamma[...,None])*n*pdotn[...,None]
    return pnew

def myantenna(p1,p2,xi_m,gen,debug=False):
    """
    Algorithm 1 (BASIC ANTENNA) from hep-ph/0004047, with massless inputs.

    p1, p2: (...,3) spatial momenta of massless particles (E=|p|).
    xi_m:  scalar cutoff (float or tensor broadcastable to (...,)).
    returns k: (...,3) spatial momentum of a massless particle (E=|k| implicitly).
    """

    NEv = p1.shape[0]

    #1. boost to CM frame and get kinematics of p1 and p2
    E1, E2 = p1.norm(dim=-1), p2.norm(dim=-1)
    ECM = torch.sqrt(2*(E1*E2 - (p1*p2).sum(-1)))
    if debug: print('ECM = ',ECM)
    if torch.any(torch.isnan(ECM)):
        raise ValueError("ECM^2 is negative")
    pCMnorm = (p1+p2).norm(dim=-1)
    nCM = (p1+p2)/(pCMnorm[...,None]+1e-20) # unit vector of boost from CM to original frame; regulate denominator if already in CM frame
    yCM = torch.arccosh((E1+E2)/ECM) # positive by construction

    p1CM = myboost_massless(p1,nCM,yCM,sign=-1)
    if debug:
        p2CM = myboost_massless(p2,nCM,yCM,sign=-1)
        print('nCM = ',nCM, 'yCM = ',yCM)
        print('p1+p2 in CM frame:',p1CM+p2CM)

    p1cth = p1CM[...,2]/p1CM.norm(dim=-1)
    p1sth = torch.sqrt(1-p1cth*p1cth)
    if torch.any(torch.isnan(p1sth)):
        print(p1cth[torch.isnan(p1sth)])
        raise ValueError("problem with p1 theta")
    p1phi = torch.atan2(p1CM[...,1],p1CM[...,0])
    p1cphi = torch.cos(p1phi)
    p1sphi = torch.sin(p1phi)
    if debug: print('p1 cos theta =',p1cth, 'p1 cos phi = ', p1cphi)

    #2. sample log-uniform variables in float-64
    lx = math.log(xi_m)
    u = torch.rand((NEv, 2), dtype=torch.float64, generator=gen)
    xi = torch.exp((2.0*u - 1.0)*lx)
    xi1, xi2 = xi[..., 0], xi[..., 1]

    #4. sample phi in frame of p1
    phi = 2*math.pi*torch.rand(NEv, dtype=torch.float64, generator=gen)

    #if debug, fix values for the random variables
    if debug:
        xi1 = 2.0*torch.ones(NEv)
        xi2 = 0.2*torch.ones(NEv)
        phi = 0.0*torch.ones(NEv)

    #3. convert to k0 and cos theta
    k0 = 0.5*ECM*(xi1+xi2)
    cth = (xi2-xi1)/(xi1+xi2)
    sth = torch.sqrt(1-cth*cth)
    if debug: print('k0 = ',k0,'cth = ',cth,'sth = ',sth)
    if torch.any(torch.isnan(sth)):
        raise ValueError("problem with k theta")

    #5. construct k
    k = torch.vstack((k0*sth*torch.cos(phi),k0*sth*torch.sin(phi),k0*cth)).T
    if debug: print('k = ',k)

    #6. rotate from frame where p1 is along z-axis to original frame, then boost back

    k_rotated_x = k[...,0]*p1cth*p1cphi + k[...,2]*p1sth*p1cphi - k[...,1]*p1sphi
    k_rotated_y = k[...,1]*p1cphi + k[...,0]*p1cth*p1sphi + k[...,2]*p1sth*p1sphi
    k_rotated_z = k[...,2]*p1cth - k[...,0]*p1sth
    k_rotated = torch.vstack((k_rotated_x,k_rotated_y,k_rotated_z)).T
    if debug: 
        ex = torch.Tensor([p1cth*p1cphi, p1cth*p1sphi, -p1sth])/torch.Tensor([p1cth*p1cphi, p1cth*p1sphi, -p1sth]).norm(dim=-1)
        ey = torch.Tensor([-p1sphi, p1cphi, 0.0])/torch.Tensor([-p1sphi, p1cphi, 0.0]).norm(dim=-1)
        print('ex = ',ex,'ey = ', ey)
        print('ex x ey',torch.cross(ex,ey,dim=-1), 'ez = ',p1CM/p1CM.norm(dim=-1))
        print('k_rotated =',k_rotated)
        print('norm ratio =',k_rotated.norm(dim=-1)/k.norm(dim=-1))
        
    k_boost = myboost_massless(k_rotated,nCM,yCM)

    return k_boost

def myQCD_antenna(nevents, nparticles, xi_m=20.0, energy=1.0):
    gen = torch.Generator(); gen.seed()

    #1. generate q1 and qn in CM frame

    q1cth = -1 + 2*torch.rand(nevents,dtype=torch.float64, generator=gen)
    q1sth = torch.sqrt(1-q1cth*q1cth)
    q1phi = 2*math.pi*torch.rand(nevents, dtype=torch.float64, generator=gen)
    q1cphi = torch.cos(q1phi)
    q1sphi = torch.sin(q1phi)

    qs = torch.empty((nevents, nparticles, 3), dtype=torch.float64)
    qs[:,0,0], qs[:,0,1], qs[:,0,2] = (energy/2)*q1sth*q1cphi, (energy/2)*q1sth*q1sphi, (energy/2)*q1cth
    qs[:,-1]= -qs[:,0]

    #2. loop over basic antennas

    for i in range(1, nparticles - 1):
        qs[:, i] = myantenna(qs[:, i-1], qs[:, -1], xi_m, gen)

    #3-4. apply the RAMBO map
    ps = qs_to_ps(qs, energy)

    #Algorithm 3: symmetrize
    perm = torch.rand(nevents, nparticles).argsort(dim=1) #trick to apply a random permutation for across particles for each event
    return ps[torch.arange(nevents)[:, None], perm]

def main(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("--n-events", type=int, default=100000)
	p.add_argument("--n-particles", type=int, default=10)
	p.add_argument("--xi", type=float, default=20.0)
	a = p.parse_args(argv)

	events = myQCD_antenna(a.n_events,a.n_particles,xi_m = a.xi)
	torch.save(events.cpu(),f"SARGE_N{a.n_particles}_xi{a.xi}.pt")

if __name__ == "__main__":
    main()
