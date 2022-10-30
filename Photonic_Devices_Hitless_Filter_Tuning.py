import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import pi, ceil, sqrt

# Filter requirements
lambda0 = 1.55e-6
N_channel = 8
channel_separation = 100e9
FSR = N_channel*channel_separation # Free spectral range
B = 40e9 # Bandwidth

# Physical Constants
c = 3e8
n_Si = 3.45
n_Si02 = 1.45
n_eff = 2.356943162
n_g = 3.5
gamma = 1
R_min = 5*((n_Si - n_Si02)/n_Si02)**(-1.5) * 1e-6

# Filter design
finesse = FSR/B * 1e1
K_r = pi/finesse
# gamma = 1 - K_r
# L_r = c/(n_eff*FSR)
# N = floor(n_eff*L_r/lambda0) # Interference order
N = ceil(2*pi*R_min*n_eff/lambda0)
R = lambda0/(n_eff*2*pi) * N
L_r = 2*pi*R

FSR = c/(n_eff*L_r)

f0 = c/lambda0

def H_d(f_norm, K_r):
    f = f0 + f_norm*FSR
    t = np.sqrt(K_r)
    r = np.sqrt(1-K_r)
    beta = (2*pi/c)*n_eff*f
    
    return -(t**2*sqrt(gamma)*np.exp(-1j*beta*L_r/2))/(1 - r**2*gamma*np.exp(-1j*beta*L_r))

def H_t(f_norm, K_r):
    f = f0 + f_norm*FSR
    t = np.sqrt(K_r)
    r = np.sqrt(1-K_r)
    beta = (2*pi/c)*n_eff*f

    return r*(1 - gamma*np.exp(-1j*beta*L_r))/(1 - r**2*gamma*np.exp(-1j*beta*L_r))


# Report
print("R_min: {:e}".format(R_min));
print("R: {:e}".format(R));
print("FSR: {:e}".format(FSR));
print("N: {}".format(N));
print("Finesse: {:e}".format(finesse));
print("K_r: {:e}".format(K_r));
for i in range(0, 9):
    print("Cross-talk Channel {}: {}".format(i,10*np.log10(np.abs(H_d(i*channel_separation/FSR, K_r)**2))))

# Graphics
# Transfer functions
f_norm = np.linspace(-0.25, 1.25, 2**14)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.semilogy(f_norm, np.abs(H_d(f_norm, K_r)**2), label = r"$H_d$")
ax.semilogy(f_norm, np.abs(H_t(f_norm, K_r)**2), label = r"$H_t$")
for i in range(1, 4):
    ax.vlines(i*channel_separation/FSR, 0, 1)
ax.legend()
ax.set_xlabel(r"$\frac{f}{FSR}$")
ax.set_ylabel("H [db]")

# Cross-talk
K_r = np.linspace(0, 0.5, 128)
ax_cross_talk = fig.add_subplot(1, 2, 2)
for i in range(1, 4):
    ax_cross_talk.semilogy(K_r, np.abs(H_d(i*channel_separation/FSR, K_r)**2), label = r"$Channel {}$".format(i))
ax_cross_talk.legend()
ax_cross_talk.set_xlabel(r"$K_r$")
ax_cross_talk.set_ylabel("Cross-talk [db]")

# Hitless filter

def H_d_hitless(f_norm, K_c, phi):
    f = f0 + f_norm*FSR
    nu = 4*K_c*(1-K_c)
    T_MZI = nu/2*(1+np.cos(phi))
    t = np.sqrt(T_MZI)
    r = np.sqrt(1-T_MZI)
    beta = (2*pi/c)*n_eff*f
    
    return -(t**2*sqrt(gamma)*np.exp(-1j*beta*L_r/2))/(1 - r**2*gamma*np.exp(-1j*beta*L_r))

fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)

phi = np.linspace(0, 2*pi, 2**14)
ax.semilogy(phi, np.abs(H_d_hitless(0, 0.001, phi)**2), label = r"$H_d$")
ax.legend()
ax.set_xlabel(r"$\frac{f}{FSR}$")
ax.set_ylabel("H [db]")

plt.show()