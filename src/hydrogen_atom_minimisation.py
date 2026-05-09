import numpy as np
import matplotlib.pyplot as plt

def metropolis_nd(x0, func, N, sigma=1, burn=5):
    n = len(x0)
    x = [x0]
    P = func(x0)
    for L in range(N+burn+2):
        target = x[-1] + np.random.normal(0, sigma, size=n)
        P_p = func(target)
        if np.random.uniform()<min(P_p/P, 1):
            x.append(target)
            P = P_p
        else:
            x.append(x[-1])
    return x[burn+2:]

def finite_difference_nd(func, x, step):
    d = len(x)
    total = -2*d*func(x)
    step_vec = np.zeros(d)
    for i in range(d):
        step_vec[i] = step
        total += func(x+step_vec)+func(x-step_vec)
        step_vec[i] = 0
    return total/(step*step)

def wavefunc(x, theta):
    return np.exp(-theta*np.linalg.norm(x))

def pdf(x, theta):
    return wavefunc(x, theta)**2

def energy(theta, n_batches, n_chains):
    points = []
    for i in range(n_chains):
        points += metropolis_nd(np.random.normal(0, 1, size=3), lambda x:pdf(x, theta), n_batches)
    energy_points = []
    for point in points:
        energy_points.append(-.5*finite_difference_nd(lambda x:wavefunc(x, theta), point, 1e-6)/wavefunc(point, theta)-1/np.linalg.norm(point))
    mean, std = np.mean(energy_points), np.std(energy_points)/np.sqrt(n_batches*n_chains)
    points, energy_points = np.array(points), np.array(energy_points)
    grads = -np.linalg.norm(points, axis=1) * (energy_points - mean)
    return mean, std, 2*np.mean(grads), 2*np.std(grads)/np.sqrt(n_batches*n_chains)

def gradient_descent(theta_0, alpha):
    E = energy(theta_0, 5, 100)
    grad_0 = E[2]
    theta = theta_0
    grad = grad_0
    while (abs(grad)+E[3])*alpha > 1e-6:
        theta = theta - grad*alpha
        E = energy(theta, 5, 10+int(1/abs(grad)))
        print(10+int(1/abs(grad)))
        grad = E[2]
        print(theta, grad*alpha, E[3]*alpha)
    return theta, abs(grad*alpha), E[0]

theta, grjnjneje, energy = gradient_descent(100, 0.5)
print(theta, energy)

# points = []
# for i in range(5):
#     points += metropolis_nd(np.random.normal(0, 1, size=3), lambda x:pdf(x, 1), 100000000)
# points = np.transpose(np.array(points))

# plt.figure(figsize=(6,6))
# plt.hist2d(points[0], points[1], bins=20, range=[[-2, 2],[-2,2]])
# plt.savefig("heatmap H1 1")
# plt.show()
# plt.figure(figsize=(6,6))
# plt.hist2d(points[0], points[1], bins=50, range=[[-2, 2],[-2,2]])
# plt.savefig("heatmap H1 2")
# plt.show()
# plt.figure(figsize=(6,6))
# plt.hist2d(points[0], points[1], bins=100, range=[[-2, 2],[-2,2]])
# plt.savefig("heatmap H1 3")
# plt.show()
# plt.figure(figsize=(6,6))
# plt.hist2d(points[0], points[1], bins=200, range=[[-2, 2],[-2,2]])
# plt.savefig("heatmap H1 4")
# plt.show()

# thetas = np.linspace(0.95,1.05,100,endpoint=False)

# points = []

# for i in range(10):
#     points += metropolis_nd(np.random.normal(0, 1, size=3), lambda x:pdf(x, 1), 10000)

# energies_theta = []
# for theta in thetas:
#     points_theta = np.array(points)/theta
#     energy_points = []
#     for point in points_theta:
#         energy_points.append(-.5*finite_difference_nd(lambda x:wavefunc(x, theta), point, 1e-6)/wavefunc(point, theta)-1/np.linalg.norm(point))
#     energies_theta.append(np.mean(energy_points))

# plt.scatter(thetas, energies_theta)
# plt.show()