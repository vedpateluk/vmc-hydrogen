import numpy as np
import matplotlib.pyplot as plt

def finite_difference_sampler(N, func, step, start): # 2nd order
    x = np.linspace(0, N*step, N) + start - step
    y = func(x)
    y_pp = []
    for i in range(1,N-1):
        y_pp.append((y[i+1]+y[i-1]-2*y[i])/(step*step))
    return x, np.array(y_pp)

def finite_difference_2nd_order(func, x, step):
    return (func(x+step)+func(x-step)-2*func(x))/(step*step)

def finite_difference_4th_order(func, x, step):
    return (-func(x+2*step)+16*func(x+step)-30*func(x)+16*func(x-step)-func(x-2*step))/(12*step*step)

def finite_difference_6th_order(func,x,step):
    return (-func(x+3*step)+9*func(x+2*step)-45*func(x+step)+45*func(x-step)-9*func(x-2*step)+func(x-3*step))/(60*step**2)

def sample(N, func, step, start):
    x = np.linspace(0, N*step, N) + start
    return x, func(x)

# N_points = 100

# x, y = sample(N_points, np.exp, 10/N_points, 0)
# y_n = finite_difference_sampler(N_points, np.exp, 10/N_points, 0)[1]
# x = x[1:-1]
# y = y[1:-1]

# plt.plot(x, y_n-y)
# plt.plot(x, y*0+0.1**2)
# plt.show()

N = np.int32(np.exp(np.linspace(1,13,100)))
step = 10/N

fd_2, fd_4, fd_6 = [], [], []

for n in N:
    s = 10/n
    f_2, f_4, f_6 = [], [], []
    x = 1
    for i in range(n):
        f_2.append((finite_difference_2nd_order(np.exp, x, s)-np.exp(x))**2)
        f_4.append((finite_difference_4th_order(np.exp, x, s)-np.exp(x))**2)
        f_6.append((finite_difference_6th_order(np.exp, x, s)-np.exp(x))**2)
        x+=s
    fd_2.append(np.sqrt(sum(f_2)/n))
    fd_4.append(np.sqrt(sum(f_4)/n))
    fd_6.append(np.sqrt(sum(f_6)/n))

fd_2, fd_4, fd_6 = np.array(fd_2), np.array(fd_4), np.array(fd_6)

plt.plot(step, np.abs(fd_2), label="2nd Order")
plt.plot(step, np.abs(fd_4), label="4th Order")
plt.plot(step, np.abs(fd_6), label="6th Order")
plt.xscale('log')
plt.yscale('log')
plt.title("Finite Difference Absolute Error against Step Size")
plt.xlabel("step size (log scale)")
plt.ylabel("absolute error (log scale)")
plt.legend()
plt.savefig("1d_finite_difference_step_size")
plt.show()

