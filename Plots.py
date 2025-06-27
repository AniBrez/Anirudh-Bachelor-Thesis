#%%
from cmath import phase

#Imports
import numpy as np
from matplotlib.markers import MarkerStyle
from numpy.lib.format import write_array_header_1_0
from phaseportrait.trajectories import trajectory
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import gridspec
from statistics import mean
from matplotlib import cm, colors
# import matplotlib as mpl
# mpl.use("pdf")
# mpl.rcParams.update({
#     "font.family": "serif",
#     "text.usetex": True,
#     "pdf.fonttype": 42,
#     'axes.titlesize': 16,
#     'axes.labelsize': 14,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 12
# })
#%% md
# LaTeX font conversion:
#%%
# plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": [],  # Let LaTeX handle it
#     "axes.labelsize": 12,
#     "font.size": 12,
#     "legend.fontsize": 10,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "text.latex.preamble": r"""
#         \usepackage{tgschola}
#         \usepackage[T1]{fontenc}
#         \usepackage{amsmath}
#     """,
#     "savefig.format": "pgf"
# })
#%% md
# Single neuron time-series
#%%
def neuron(y,t,I):
    theta = y
    dydt = (1-np.cos(y)) + (1+np.cos(y))*I
    return dydt

I = 0.5
y0 = 0
t = np.linspace(0,20,100)
sol = odeint(neuron, y0, t, args=(I,))
sol_mod = np.mod(sol[:,0], 2*np.pi)

#plot
plt.figure(figsize=(6, 2)) 
plt.plot(t, sol_mod,color='purple')
plt.xlabel(r'$t$')
plt.ylabel(r'$\theta$')
plt.yticks([0, np.pi/2, np.pi,3*np.pi/2,2*np.pi], [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{2}$', r'$2\pi$'])
plt.xticks(np.arange(0, 21, 2))
plt.savefig("spiking.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
#%% md
# Two neuron model with adaptive coupling
#%%
def neuron_adaptive(y,t,I,a):
    theta_1, theta_2 = y
    dydt = [(1-np.cos(theta_1)) + (1+np.cos(theta_1))*I, (1-np.cos(theta_2)) + (1+np.cos(theta_2))*(I-(6/(theta_2-np.pi)**2)*np.sin(theta_2-theta_1-a))]
    return dydt

#(2/(theta_2-np.pi)**2)
y0 = [0,np.pi/2]
I=0.1
a=3*np.pi/2
sol_adapt = odeint(neuron_adaptive, y0, t, args=(I,a,))
sol_mod_1 = np.mod(sol_adapt[:,0], 2*np.pi)
sol_mod_2 = np.mod(sol_adapt[:,1], 2*np.pi)


#plot
plt.figure(figsize=(15, 6)) 
plt.plot(t, sol_mod_1,'b', label='neuron 1')
plt.plot(t, sol_mod_2,'r', label='neuron 2')
plt.xlabel('time [t]')
plt.ylabel('phase' + r' [$\theta$]')
plt.title('Coupled neuron model')
#plt.savefig("phaselocking.pgf")
#%% md
# Two neuron model error dynamics (NOT RELEVANT) 
#%% md
# 
#%%
def adaptive_neuron_error(y,t,I):
    theta, e = y
    dydt = [(1-np.cos(theta)) + (1+np.cos(theta))*(1+np.tanh(e)), (1-I)*(np.cos(theta-e)-np.cos(theta)) + (1+np.cos(theta))*(I-4*(1/(1+np.cos(theta)))*e - e**2)]
    return dydt

t = np.linspace(0,100,1000)
y0 = [0,I]
sol = odeint(adaptive_neuron_error, y0, t, args=(I,))
sol_err = sol[:,1]

#plot
plt.figure(figsize=(15, 6)) 
plt.plot(t, sol_err,'r')
plt.xlabel('time [t]')
plt.ylabel('error' + r' [$e$]')
plt.title('Error dynamics')
#%%
# Define the system
def phase_sys(x, y, k, I_1, I_2):
    dx = (1 - np.cos(x)) + (1 + np.cos(x)) * I_1
    dy = (1 - np.cos(y)) + (1 + np.cos(y)) * (I_2 + k*np.sin(x-y))
    return dx, dy

# Create grid
x_vals = np.linspace(0, 2 * np.pi, 100)
y_vals = np.linspace(0, 2 * np.pi, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the phase system
def plot_phase_sys(ax, phase_sys, X, Y, k, I_1, I_2):
    DX, DY = phase_sys(X, Y, k, I_1, I_2)
    speed = np.sqrt(DX**2 + DY**2)
    DXn = DX / speed
    DYn = DY / speed

    # Plot with streamlines
    strm = ax.streamplot(X, Y, DXn, DYn, color=speed, cmap='plasma', linewidth=1.2, density=1.2,arrowsize=0.5)
    colorbar = plt.colorbar(strm.lines)
    colorbar.set_label(r'$||\omega||_2$',labelpad=10)
    colorbar.set_ticks([])
    # Add labels with LaTeX
    # Kuramoto model
    # ax.set_xlabel(r'$\phi_1$')
    # ax.set_ylabel(r'$\phi_2$')
    # Theta neuron model
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')

    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
               [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
               [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax.grid(True)
    
#plt.savefig("homo_synch_pp.pdf")
# plot_phase_sys(plt,X,Y,0.5,0.5)
# plt.show()
#%%
def system(Y, t, I_1, I_2, k):
    x,y=Y
    dxdt = (1 - np.cos(x)) + (1 + np.cos(x)) * I_1
    dydt = (1 - np.cos(y)) + (1 + np.cos(y)) * (I_2 + k*np.sin(x-y))
    return [dxdt,dydt]
def plot_err_sys(ax,system, Y0, k, I_1, I_2):
    t = np.linspace(0, 200, 2000)
    Y = odeint(system, Y0, t, args=(I_1,I_2,k))
    error = Y[:,0]-Y[:,1]
    ax.plot(t,error,color='red')
    ax.plot(t,0*t+mean(error),linestyle='--',color='black')
    ax.set_ylim(mean(error)-0.5, mean(error)+0.5)
    ax.set_yticks([round(mean(error),1)])
    ax.tick_params(axis='y', pad=10) 
    # ax.set_ylabel(r'$\xi$')
    ax.set_ylabel(r'$\Xi$')
    ax.set_xlabel(r'$t$')
Y0 = [np.pi/4,0]
#%%
I_1 = 0.2
I_2 = 0.7
k = 4

fig = plt.figure(figsize=(6, 6))
# First subplot: full width, top half of figure
ax1 = fig.add_axes([0.1, 0.55, 0.65, 0.5])  # [left, bottom, width, height]
# Second subplot: narrower width, bottom part of figure
ax2 = fig.add_axes([0.1, 0.32, 0.52, 0.1])  # centered narrower plot
# Plotting
plot_phase_sys(ax1,phase_sys, X, Y, k, I_1, I_2)
plot_err_sys(ax2, system, Y0, k, I_1, I_2)
plt.savefig('hetesynchk4', bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()
#%% md
# 
#%% md
# Kuramoto model 
#%%
# Create grid
x_vals = np.linspace(0, 2 * np.pi, 100)
y_vals = np.linspace(0, 2 * np.pi, 100)
X, Y = np.meshgrid(x_vals, y_vals)

def kuramoto_phase_sys(x, y, w_1, w_2):
    dx = w_1
    dy = w_2 + np.sin(x - y)
    return dx,dy

def kuramoto_sys(Y,t,w_1,w_2):
    x, y = Y
    dxdt = w_1
    dydt = w_2 + np.sin(x - y)
    return [dxdt, dydt]

w_1 = 0.5
w_2 = 0.5

fig = plt.figure(figsize=(6, 6))
# First subplot: full width, top half of figure
ax1 = fig.add_axes([0.1, 0.55, 0.65, 0.5])  # [left, bottom, width, height]
# Second subplot: narrower width, bottom part of figure
ax2 = fig.add_axes([0.1, 0.32, 0.52, 0.1])  # centered narrower plot
# Plotting
plot_phase_sys(ax1,kuramoto_phase_sys, X, Y, w_1, w_2)
plot_err_sys(ax2,kuramoto_sys, Y0, w_1, w_2)
plt.savefig('kuramoto_synch.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

#%% md
# SNLC bifurcation
#%%
def f(x, I):
    return (1 - np.cos(x)) + (1 + np.cos(x)) * I

x_vals = np.linspace(-np.pi, np.pi, 100)
y_vals = f(x_vals, -0.5)

zeros = [i for i in x_vals if abs(f(i,-0.5)) < 1e-2]
print(f'the zeros are: {zeros[0]}, {zeros[1]}')
plt.figure(figsize=(5, 2))
ax = plt.gca()
plt.plot(x_vals, y_vals, label=r'$\dot \theta$', color='purple')
plt.plot(x_vals, np.zeros_like(x_vals), 'k-', linewidth=0.8, label='zero line')
marker_stable = MarkerStyle('d', 'top')
marker_asymstable = MarkerStyle('d')
plt.plot(zeros[1], 0,
         marker=marker_asymstable,
         markersize=10,
         markerfacecolor='white',     # Fill color
         markeredgecolor='black',     # Border color
         markeredgewidth=1,           # Border width
         linestyle='',                # No connecting line
         zorder=5)
plt.plot(zeros[0], 0,
         marker=marker_asymstable,
         markersize=10,
         markerfacecolor='black',     # Fill color
         markeredgecolor='black',     # Border color
         markeredgewidth=1,           # Border width
         linestyle='',                # No connecting line
         zorder=5)
plt.arrow(-np.pi,0,np.pi/4,0, head_width=0.3, head_length=0.15, linewidth=0.0001, fc='black', ec='black')
plt.arrow(zeros[1],0,-zeros[1],0, head_width=0.3, head_length=0.15, linewidth=0.0001, fc='black', ec='black')
plt.arrow(zeros[1],0,1.3*np.pi/4,0, head_width=0.3, head_length=0.15, linewidth=0.0001, fc='black', ec='black')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot \theta$')
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)
plt.xticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])
plt.yticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$\pi$'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#plt.savefig("inhibitthetaplot.pgf")

#%% md
# Phase portraits and error plots
#%%

#%%
def phase_sys_adapt(x, y, I_1,I_2,l,s,a):
    dx = (1 - np.cos(x)) + (1 + np.cos(x)) * I_1
    dy = (1 - np.cos(x)) + (1 + np.cos(x))*(I_2+(s/l)*(x-y-a)*x)
    return dx,dy

# Create grid
x_vals = np.linspace(0, 2*np.pi, 100)
y_vals = np.linspace(0, 2*np.pi, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the phase system
def plot_phase_sys_adapt(ax, X, Y, I_1, I_2,l,s,a):
    DX, DY = phase_sys_adapt(X, Y, I_1, I_2,l,s,a)
    speed = np.sqrt(DX**2 + DY**2)
    DXn = DX / speed
    DYn = DY / speed

    # Plot with streamlines
    strm = ax.streamplot(X, Y, DXn, DYn, color=speed, cmap='plasma', linewidth=1.2, density=1.2)
    colorbar = plt.colorbar(strm.lines)
    colorbar.set_ticks([])
    colorbar.set_label(r'$||\omega||_2$',labelpad=10)
    # Add labels with LaTeX
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
               [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
               [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax.grid(True)
    
#plt.savefig("homo_synch_pp.pdf")
# plot_phase_sys(plt,X,Y,0.5,0.5)
# plt.show()
Y0 = [np.pi/2,0,0]
#%%
def system_adapt(Y,t, I_1,I_2,l,s,a):
    x,y,z=Y
    dxdt = (1 - np.cos(x)) + (1 + np.cos(x)) * I_1
    dydt = (1 - np.cos(y)) + (1 + np.cos(y)) * (I_2 + z*x)
    dzdt = (1/0.1)*(-l*z+s*(x-y-a))
    return [dxdt,dydt,dzdt]
def plot_err_sys_adapt(ax,Y0,I_1,I_2,l,s,a):
    t = np.linspace(0, 200, 2000)
    Y = odeint(system_adapt, Y0, t, args=(I_1,I_2,l,s,a))
    error = Y[:,0]-Y[:,1]
    ax.plot(t,error,color='red')
    ax.plot(t,0*t+mean(error),linestyle='--',color='black')
    ax.set_ylim(mean(error)-1, mean(error)+1)
    ax.set_yticks([round(mean(error),1)])
    # ax.set_yticks([round(min(error[50:151]),1), round(mean(error),1), round(max(error[50:151]),1)])
    ax.tick_params(axis='y', pad=10) 
    ax.set_ylabel(r'$\Xi$')
    ax.set_xlabel(r'$t$')

#%%
I_1 = 0.2
I_2 = 0.7
l= 0.5
s=1
a=np.pi/4

fig = plt.figure(figsize=(6, 6))
# First subplot: full width, top half of figure
ax1 = fig.add_axes([0.1, 0.55, 0.65, 0.5])  # [left, bottom, width, height]
# Second subplot: narrower width, bottom part of figure
ax2 = fig.add_axes([0.1, 0.32, 0.52, 0.1])  # centered narrower plot
# Plotting
plot_phase_sys_adapt(ax1, X, Y, I_1, I_2,l,s,a)
plot_err_sys_adapt(ax2, Y0, I_1, I_2,l,s,a)
plt.savefig('varphipi4.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()
#%% md
# NOT RELEVANT
#%%
# I_1 = 0.2
# I_2 = 0.7
# Y0 = [np.pi/2,0,0]
# a = np.pi/4
# 
# S_list = np.linspace(0,2,10)
# L_list = np.linspace(0,2,10) 
# S,L = np.meshgrid(S_list, L_list)
# 
# error_list = []
# for s in S_list:
#     for l in L_list:
#         t = np.linspace(0, 200, 2000)
#         Y = odeint(system_adapt, Y0, t, args=(I_1,I_2,l,s,a))
#         error = np.abs(Y[:,0]-Y[:,1]-a)
#         error_list.append(np.min(error))
#     
# E = np.array(error_list).reshape(S.shape)
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(S, L, E, cmap='viridis', edgecolor='none')
# plt.show()
    