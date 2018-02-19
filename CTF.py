import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

pixel_size = 1.7
k = np.linspace(1/100,1/(2*pixel_size),1000)
Cs = 2
defocus = -4

ht = 300000
wavelength = 1.23 / ht**(1/2)
A = 0.05
sf = 1 #scaling factor
B = 150


#proper functions
ctf = -2*np.pi*(Cs * wavelength**3 * (k**4)/4 - np.pi * defocus*10000 * wavelength * (k**2)/2)
CTF = -sf * ((1-A**2)**0.5*np.sin(ctf) + A*np.cos(ctf))

#guassian envelope funtion
env = np.exp(-B*k**2)


plt.plot(k,env*CTF)
plt.ylabel('Signal')
plt.xlabel('1/A')
plt.show()

# def ctf_funtion():
#     return lambda x, y: x**2+y**2

#eq = -np.sin(np.pi/2 * Cs * wavelength**3 * k**4 + np.pi * defocus_a * wavelength * k**2) * np.exp(-0.5*(np.pi*wavelength*d)**2*k**4)

#OLD equations
#ctf_evelope = ctf_phase*envelope_c*envelope_s
#R = np.exp(-0.5*(np.pi*wavelength*d)**2*x**4) + np.exp(-0.5*(np.pi*wavelength*d)**2*y**4)
#envelope_c = np.exp(-0.5*(np.pi*wavelength*d)**2*k**4)
#envelope for the temporal envelope function
d= 10**3.5
#envelope for angular spread of the source
alpha = 1E-6
#envelope_s = np.exp(-(np.pi*alpha/wavelength)**2 * (Cs * wavelength**3*k**3 + defocus_a*wavelength*k)**2)
#t_env = np.sin((2*np.pi / wavelength) * ctf_phase)
#t_env2 = (A * ctf_amp) - np.sqrt(1-A**2)*ctf_phase

