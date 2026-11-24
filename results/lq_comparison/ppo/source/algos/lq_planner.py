"""Compiled numerical kernel for TD-MPC2-LQ. No plant information is used.

Random inputs are supplied by the caller: compiling changes arithmetic only.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def quad(w, x, u):
    return w[0]*x*x+2*w[1]*x*u+w[2]*u*u+w[3]


@njit(cache=True)
def kernel(p, warm, x, normals, uniform, H, C, E, P, I, gamma, bound, training):
    mean = np.zeros(H)
    mean[:-1] = warm[1:]
    std = np.full(H, 2.)
    prior = np.empty((P, H))
    at = 0
    sigma = np.exp(p[15])
    for j in range(P):
        z = x
        for h in range(H):
            u = min(bound, max(-bound, -p[14]*z+sigma*normals[at])); at += 1
            prior[j,h] = u
            z = p[0]*z+p[1]*u
    q = .5*(p[6:10]+p[10:14])
    actions = np.empty((C,H)); score = np.empty(C)
    elite = np.empty((E,H)); weights = np.empty(E)
    for iteration in range(I):
        for j in range(C):
            z = x; s = 0.
            for h in range(H):
                a = min(bound,max(-bound,mean[h]+std[h]*normals[at])); at += 1
                if j < P: a = prior[j,h]
                actions[j,h] = a
                s += gamma**h*quad(p[2:6],z,a)
                z = p[0]*z+p[1]*a
            a = min(bound,max(-bound,-p[14]*z+sigma*normals[at])); at += 1
            score[j] = s+gamma**H*quad(q,z,a)
        idx = np.argsort(score)[:E]
        smallest = score[idx[0]]
        total = 0.
        for j in range(E):
            elite[j] = actions[idx[j]]
            weights[j] = np.exp(-.5*(score[idx[j]]-smallest))
            total += weights[j]
        weights /= total
        for h in range(H):
            mean[h] = np.sum(elite[:,h]*weights)
            std[h] = min(2., max(.05, np.sqrt(np.sum((elite[:,h]-mean[h])**2*weights))))
    chosen = E-1; acc = 0.
    for j in range(E):
        acc += weights[j]
        if uniform < acc:
            chosen = j; break
    a = elite[chosen,0]
    if training: a += std[0]*normals[at]
    return min(bound,max(-bound,a)), mean


def plan(learner, x, rngs=None, reset=False, training=False):
    c = learner.cfg
    if reset: learner.mean[:] = 0
    rngs = rngs if rngs is not None else [r[1] for r in learner.rng]
    n = c.prior_candidates*c.horizon+c.iterations*c.candidates*(c.horizon+1)+1
    result = np.empty(len(x))
    for j, rng in enumerate(rngs):
        result[j], learner.mean[j] = kernel(
            learner.p[j], learner.mean[j], x[j], rng.standard_normal(n), rng.random(),
            c.horizon,c.candidates,c.elites,c.prior_candidates,c.iterations,
            c.gamma,c.bound,training)
    return result
