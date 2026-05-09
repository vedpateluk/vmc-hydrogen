import numpy as np
import matplotlib.pyplot as plt

def metropolis_nd(x0, func, N, sigma=1, burn=5):
    x = [x0]
    P = func(x0)
    for L in range(N+burn+2):
        target = x[-1] + np.random.normal(0, sigma, size=(2, 3))
        P_p = func(target)
        if np.random.uniform()<min(P_p/P, 1):
            x.append(target)
            P = P_p
        else:
            x.append(x[-1])
    return x[burn+2:]

def finite_difference_nd(func, x, step):
    x_flat = x.flatten()
    d = len(x_flat)
    total = -2*d*func(x)
    step_flat = np.zeros(d)
    for i in range(d):
        step_flat[i] = step
        step_vec = step_flat.reshape(2,3)
        total += func(x+step_vec)+func(x-step_vec)
        step_flat[i] = 0
    return total/(step*step)

def wavefunc(x, q, theta):
    t_1 = np.exp(-theta[0]*(np.linalg.norm(x[0]-q[0]) + np.linalg.norm(x[1]-q[1])))
    t_2 = np.exp(-theta[0]*(np.linalg.norm(x[0]-q[1]) + np.linalg.norm(x[1]-q[0])))
    return (t_1+t_2) * np.exp(-theta[1]/(1+theta[2]*np.linalg.norm(x[0]-x[1])))

def pdf(x, q, theta):
    return wavefunc(x, q, theta)**2

def local_energy(x, q, theta):
    kinetic = -.5*finite_difference_nd(lambda x_:wavefunc(x_, q, theta), x, 1e-6)
    potential = 1/np.linalg.norm(x[0]-x[1]) + 1/np.linalg.norm(q[0]-q[1])
    for i in [0,1]:
        for j in [0,1]:
            potential += 1/np.linalg.norm(x[i]-q[j])
    return kinetic/wavefunc(x, q, theta) + potential