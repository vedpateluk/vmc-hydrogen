import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

n_points = 0

def metropolis_nd(x0, func, N, sigma=1, burn=5):
    x = [x0]
    P = func(x0)
    for L in range(N+burn+2):
        target = x[-1] + np.random.normal(0, sigma, size=(2, 3))
        P_p = func(target)
        if P<P_p:
            ratio = 1
        else:
            ratio = P_p/P
        if np.random.uniform()<ratio:
            x.append(target)
            P = P_p
        else:
            x.append(x[-1])
    return x[burn+2:]

def finite_difference_nd_2nd(func, x, step):
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

def finite_difference_nd(func, x, step):
    x_flat = x.flatten()
    d = len(x_flat)
    total = -30*d*func(x)
    step_flat = np.zeros(d)
    for i in range(d):
        step_flat[i] = step
        step_vec = step_flat.reshape(2,3)
        total += -func(x+2*step_vec)+16*func(x+step_vec)+16*func(x-step_vec)-func(x-2*step_vec)
        step_flat[i] = 0
    return total/(12*step*step)

def wavefunc(x, q, theta):
    t_1 = np.exp(-theta[0]*(np.linalg.norm(x[0]-q[0]) + np.linalg.norm(x[1]-q[1])))
    t_2 = np.exp(-theta[0]*(np.linalg.norm(x[0]-q[1]) + np.linalg.norm(x[1]-q[0])))
    return (t_1+t_2) * np.exp(-theta[1]/(1+theta[2]*np.linalg.norm(x[0]-x[1])+1e-12))

def pdf(x, q=np.array([[.5,0,0],[-.5,0,0]]), theta=np.array([1,1,1])):
    return wavefunc(x, q, theta)**2

def local_energy(x, q, theta):
    kinetic = -.5*finite_difference_nd(lambda x_:wavefunc(x_, q, theta), x, 1e-6)
    potential = 1/(np.linalg.norm(x[0]-x[1])+1e-12) + 1/(np.linalg.norm(q[0]-q[1])+1e-12)
    for i in [0,1]:
        for j in [0,1]:
            potential -= 1/(np.linalg.norm(x[i]-q[j])+1e-12)
    return kinetic/wavefunc(x, q, theta) + potential

def energy(q, theta, n_batches, len_chain, std = False, gradient = False):
    # global n_points
    points = []
    for i in range(n_batches):
        points += metropolis_nd(np.random.normal(0, 1, size=(2,3)), lambda x:pdf(x, q, theta), len_chain)
    energy_points = []
    for point in points:
        energy_points.append(local_energy(point, q, theta))
        # n_points += 1
        # print(n_points)
    mean = np.mean(energy_points)
    if std and not gradient:
        return mean, np.std(energy_points)
    elif gradient:
        grads = []
        for point, energy in zip(points, energy_points):
            grad = np.zeros(3)
            A = np.linalg.norm(point[0]-q[0])+np.linalg.norm(point[1]-q[1])
            B = np.linalg.norm(point[0]-q[1])+np.linalg.norm(point[1]-q[0])
            C = np.linalg.norm(point[1]-point[0])
            t1 = np.exp(-theta[0]*A)
            t2 = np.exp(-theta[0]*B)
            grad[0] = (energy - mean) * (-A*t1-B*t2)/(t1+t2)
            grad[1] = (energy - mean) * (-1/(1+theta[2]*C))
            grad[2] = (energy - mean) * (theta[1]*C/(1+theta[2]*C)**2)
            grads.append(grad)
        grads = 2*np.array(grads)
        mean_grad, std_grad = np.mean(grads, axis=0), np.std(grads, axis=0)
        if std:
            return mean, np.std(energy_points)/np.sqrt(n_batches*len_chain), mean_grad, std_grad/np.sqrt(n_batches*len_chain)
        return mean, mean_grad
    return mean

def samples_from_grad(grad):
    return int(10 + 200/grad + .02/grad**2)

def gradient_descent(q, theta_0, alpha):
    E = energy(q, theta_0, 25, 100, gradient=True, std=True)
    print(E)
    grad_0 = E[2]
    err_grad = E[3]
    theta = theta_0
    grad = grad_0
    while ((np.linalg.norm(grad)+np.linalg.norm(err_grad))*alpha > 1e-3).any():
        theta = theta - grad*alpha
        E = energy(q, theta, 5, samples_from_grad(np.linalg.norm(grad)), gradient=True, std=True)
        print(samples_from_grad(np.linalg.norm(grad)))
        grad = E[2]
        err_grad = E[3]
        print(theta, np.linalg.norm(grad*alpha), np.linalg.norm(err_grad*alpha), E[0])
    return theta, abs(grad*alpha), E[0], E[1]

def simulated_annealing(E_func, theta_0, N, sigma=1, T_0=1):
    theta = theta_0.copy()
    E = E_func(theta)
    for i in range(N):
        T = T_0 * np.exp(-i / (1*N))
        target = theta + np.random.normal(0, sigma, size=(3))
        if not ((target>0).all() and (target<5).all()):
            continue
        E_p = E_func(target)
        if E_p-E < 0:
            ratio = 1
        else:
            ratio = np.exp(-(E_p-E)/T)
        if np.random.uniform()<ratio:
            theta = target
            E = E_p
        print(theta, E)
    return theta, E

def morse(r, D, a, r_0):
    return D * ((1-np.exp(-a*(r-r_0)))**2 - 1) - 1

print(gradient_descent(q=np.array([[-1/2,0,0],[1/2,0,0]]), theta_0=np.random.uniform(0,2,size=3), alpha=.1))

# dists = np.linspace(.5, 3, 100)
# energies = []
# err_E = []
# for dist in dists:
#     E = gradient_descent(q=np.array([[-dist/2,0,0],[dist/2,0,0]]), theta_0=np.random.uniform(0,2,size=3), alpha=.1)
#     energies.append(E[2])
#     err_E.append(E[3])
#     print(dist)

# fit, cov = op.curve_fit(morse, dists, energies, (1, 1, 1), sigma=err_E)
# x_points = np.linspace(.5, 3, 100)


# plt.errorbar(dists, energies, yerr=err_E, fmt="x")
# plt.plot(x_points, morse(x_points, fit[0], fit[1], fit[2]))
# plt.title("Morse Potential")
# plt.xlabel("Atomic separation")
# plt.ylabel("Energy")
# plt.savefig("graph")
# plt.show()

# print("distances", dists)
# print("energies", energies)
# print("fit",fit)
# fit_err = (np.sqrt(cov[0][0]), np.sqrt(cov[1][1]), np.sqrt(cov[2][2]))
# print("fit error", fit_err)
# print("fit percentage error (%)", 100*np.array(fit_err)/np.abs(np.array(fit)))


# thetas = []
# radii = np.linspace(0.1, 40, 1000)
# energies = []
# for r in radii:
#     theta = np.abs(np.random.normal(0, 1, size=3))
#     theta = theta*r/np.linalg.norm(theta)
#     thetas.append(theta)
#     energies.append(energy(np.array([[.5,0,0],[-.5,0,0]]), theta, 2, 100))
# energies = np.array(energies)
# print(len(energies), len(radii))

# plt.yscale('log')
# plt.scatter(radii, np.abs(energies))

# plt.show()

# min_thetas = []
# theta_new = (np.array([1,1,1]),1)
# for i in range(10):
#     theta_new = simulated_annealing(lambda theta:energy(np.array([[.5,0,0],[-.5,0,0]]), theta, 1, 100), theta_new[0], 100)
#     min_thetas.append(theta_new)
# print(min_thetas)
# E = gradient_descent(q=np.array([[-1,0,0],[1,0,0]]), theta_0=np.random.uniform(0,2,size=3), alpha=.1)
# print(E)

# points=[]
# for i in range(100):
#     print(i)
#     points += metropolis_nd(np.random.normal(0, 1, size=(2,3)), lambda x:pdf(x, q=np.array([[-1,0,0],[1,0,0]]), theta=np.array([1.06318451, 0.82479793, 0.75099427])), 200000)
# points = np.array(points)
# x, y = points[:,0,0], points[:,0,1]
# x, y = np.concatenate([x,points[:,1,0]]), np.concatenate([y,points[:,1,1]])


# plt.figure(figsize=(6,6))
# plt.hist2d(x, y, bins=20, range=[[-2,2],[-2,2]])
# plt.savefig("Heatmap H2 1")
# plt.show()
# plt.figure(figsize=(6,6))
# plt.hist2d(x, y, bins=50, range=[[-2,2],[-2,2]])
# plt.savefig("Heatmap H2 2")
# plt.show()
# plt.figure(figsize=(6,6))
# plt.hist2d(x, y, bins=100, range=[[-2,2],[-2,2]])
# plt.savefig("Heatmap H2 3")
# plt.show()
# plt.figure(figsize=(6,6))
# plt.hist2d(x, y, bins=200, range=[[-2,2],[-2,2]])
# plt.savefig("Heatmap H2 4")
# plt.show()