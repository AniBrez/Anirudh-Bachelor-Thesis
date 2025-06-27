#%%
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
#%%
def adapt_rule(y,t,e):
    theta = y
    dydt = np.sin(e)-theta
    return dydt
y0 = 0
t = np.linspace(0, 10, 100)
for e in [0,np.pi/2, np.pi, 3*np.pi/2]:
    y = odeint(adapt_rule, y0, t, args=(e,))
    plt.plot(t, y, label=f'e={round(e,1)}')
plt.legend(loc='best')
plt.show()
#%%
def system(Y,t,I_1,I_2,e,l,s,a):
    x, y, z = Y
    dxdt = (1-np.cos(x)) + (1+np.cos(x))*I_1
    dydt = (1-np.cos(y)) + (1+np.cos(y))*(I_2+100*np.sin(x-y))
    dzdt = (I_1-I_2)-l*z
    return [dxdt,dydt,dzdt]
Y0 = [0,np.pi/2,0]
t = np.linspace(0, 200, 2000)
I_1 = 0.8
I_2 = 0.2
e = 0.01
l = 0.5
s = 5
a=np.pi/2

Y = odeint(system, Y0, t, args=(I_1,I_2,e,l,s,a))
t_amend = t[(t>=0) & (t<=200)]
Y_amend = Y[(t>=0) & (t<=200)]
plt.plot(t_amend, np.mod(Y_amend[:, 0],2*np.pi), label=r'$\theta_1$')
plt.plot(t_amend, np.mod(Y_amend[:, 1],2*np.pi), label=r'$\theta_2$')
plt.legend(loc='best')
plt.show()
#%%
plt.plot(t,Y[:,0]-Y[:,1])
plt.plot(t,0*t+np.pi/2)
min_tot_err = min(Y[:,0]-Y[:,1]-np.pi/2)
max_tot_err = max(Y[:,0]-Y[:,1]-np.pi/2)
plt.show()
print(f"Supremum: {max_tot_err} " + f" Infimum: {min_tot_err}")
plt.plot(t,Y[:,2])
plt.show()
plt.plot(t,Y[:,2]*np.sin(Y[:,0]-Y[:,1]-np.pi/2))
plt.show()
#%%
def res_system(Y,t,I1,I2):
    x, y = Y
    dxdt = (1-np.cos(x)) + (1+np.cos(x))*I1
    dydt = (1-np.cos(y)) + (1+np.cos(y))*I2
    return [dxdt,dydt]

Y0 = [0,np.pi/2]
t = np.linspace(0,100,1000)
I1 = 0.5
I2 = 0.2
Y = odeint(res_system,Y0,t,args=(I1,I2,))
error = Y[:,1]-Y[:,0]
#plt.plot(t,np.mod(Y[:,0],2*np.pi),'r')
#plt.plot(t,np.mod(Y[:,1],2*np.pi),'g')
plt.plot(t,Y[:,0],'r')
plt.plot(t,Y[:,1],'g')
plt.show()
plt.plot(t,error)
plt.show()
#%% md
# Quadratic Integrate and Fire
#%%
def system(x,t,I):
    dxdt = x**2 + I
    return dxdt
x0 = 0
t = np.linspace(0,50,100)
I = 0.5
y = odeint(system,y0,t,args=(I,))
plt.plot(t,y)
#%%
def kuramoto_sys(Y,t,w_1,w_2):
    x, y = Y
    dxdt = w_1
    dydt = w_2 + np.sin(x - y)
    return [dxdt, dydt]

Y0 = [0, np.pi/2]
t = np.linspace(0, 100, 1000)
w_1 = 0.1
w_2 = 0.8

Y = odeint(kuramoto_sys, Y0, t, args=(w_1, w_2))
plt.plot(t, np.mod(Y[:, 0],2*np.pi), label=r'$\phi_1$')
plt.plot(t, np.mod(Y[:, 1],2*np.pi), label=r'$\phi_2$')
plt.legend(loc='best')
plt.xlabel(r'$t$')
plt.show()