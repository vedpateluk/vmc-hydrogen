import numpy as np
import matplotlib.pyplot as plt

def H(n, x):
    coeffs = [0]*n + [1]
    return np.polynomial.hermite.hermval(x, coeffs)

def metropolis(x0, func, N, sigma=1, burn=5):
    x = [x0, x0]
    P = func(x0)
    for L in range(N+burn+2):
        target = x[-1] + np.random.normal(0, sigma)
        if x[-1]!=x[-2]:
            P = func(x[-1])
        P_p = func(target)
        if np.random.uniform()<min(P_p/P, 1):
            x.append(target)
        else:
            x.append(x[-1])
    return x[burn+4:]

def finite_difference_2nd(func,x,step):
    return (func(x+step)+func(x-step)-2*func(x))/(step*step)

def finite_difference_4th(func,x,step):
    return (func(x-2*step)-16*func(x-step)+30*func(x)-16*func(x+step)+func(x+2*step))/(12*step*step)

def finite_difference_6th(func,x,step):
    return (func(x-3*step)-36*func(x-2*step)+300*func(x-step)-300*func(x+step)+36*func(x+2*step)-func(x+3*step))/(60*step*step)

def wavefunc(x):
    return np.exp(-x*x/2)*H(0,x)

def analytic_derivative(x):
    return np.exp(-x*x/2)*(x*x-1)

def pdf(x):
    return wavefunc(x)**2

# points = np.linspace(-3,3,1000)

# N = np.exp(np.linspace(-5,20,100))
# steps = 1/N
# n_points = len(points)
# errs2, errs4, errs6 = [], [], []
# for s in steps:
#     err2, err4, err6 = 0,0,0
#     for point in points:
#         err2 += (finite_difference_2nd(wavefunc, point, s)-analytic_derivative(point))**2
#         err4 += (finite_difference_4th(wavefunc, point, s)-analytic_derivative(point))**2
#         err6 += (finite_difference_6th(wavefunc, point, s)-analytic_derivative(point))**2
#     errs2.append(np.sqrt(err2/n_points))
#     errs4.append(np.sqrt(err4/n_points))
#     errs6.append(np.sqrt(err6/n_points))

# plt.plot(steps, errs2, label="2nd Order")
# plt.plot(steps, errs4, label="4th Order")
# plt.plot(steps, errs6, label="6th Order")
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.show()

points = []

for i in range(5):
    points += metropolis(np.random.normal(0, 1), pdf, 1000000)

points = np.array(points)

# plt.scatter(points, pdf(points)*np.random.rand(len(points)))
# plt.show()
counts, bins, sdffshd = plt.hist(points, bins=500, density=True, label="Normalised Sample Density")
x_points = np.linspace(bins[0],bins[-1], 100)
bins = (bins[1:]+bins[:1])/2
plt.plot(x_points, pdf(x_points)/np.sqrt(np.pi), label="Normalised PDF")
plt.title("Histogram of Samples Generated from Metropolis")
plt.xlabel("position")
plt.ylabel("density")
plt.legend()
plt.savefig("histogram_1d_gaussian")
plt.show()

# energies = -.5*finite_difference_2nd(wavefunc, points, 1e-5)/wavefunc(points)+.5*points*points

# print(np.mean(energies), np.std(energies)/np.sqrt(len(energies)))
