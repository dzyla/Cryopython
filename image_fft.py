import numpy as np
import scipy.fftpack as fp
import skimage.exposure
from scipy.optimize import curve_fit
import cryopython as cp
import matplotlib.pyplot as plt
import cv2
from skimage import util
from scipy import signal
from scipy import misc
from sklearn.preprocessing import normalize
from scipy.stats import linregress

#constants
Cs = 2
#defocus = -40000
ht = 200   #kV
A = 0.1
sf = 1 #scaling factor
B = 150
pixel_size = 1.7
wavelength = 12.398 / np.sqrt(ht * (1022 + ht))  #for kV
res_limit = 30
res_limit_max = 8

def func(x, a, b, c):
    return a*x**b+c

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def ctf_function(k,sf,defocus,offset):
    B=250
    #k = scale*k
    ctf = -2 * np.pi * (Cs * wavelength ** 3 * (k ** 4) / 4 - np.pi * defocus * wavelength * (k ** 2) / 2)
    CTF =  -sf *((1 - A ** 2) ** 0.5 * np.sin(ctf) + A * np.cos(ctf))
    env = np.exp(-B * k ** 2)
    return env*CTF+offset

#do the powerspectrum
power = cp.do_powerspectrum(cp.open_image("170715_140002.mrc"),5,0,100,5, True)
#power = cp.do_powersec_raw(cp.open_image("170715_110001.mrc"))

#find the maximum value -> should be center of the image
x0, y0 = xy = np.unravel_index(power.argmax(), power.shape)

#generate the resolution range for fitting
r = np.linspace(0,1/(2*pixel_size),int(power.shape[0]/2))

#generate the radial profile of power spectrum
prof = radial_profile(power,(x0,y0))
r_step = r[1]
r_limit_index = int((1/res_limit)/r_step)
start = r_limit_index
stop = int((1/res_limit_max)/r_step)
prof = prof[start:stop]



#scale the resolution because I don't know how to make it fixed
r_scale = lambda r: 0.9*r

#generate XY coordinates for radial fitting of CTF
X = np.linspace(-1/(2*pixel_size),1/(2*pixel_size),len(r)*2)
Y = np.linspace(-1/(2*pixel_size),1/(2*pixel_size),len(r)*2)

#Limit the resolution for fitting
r = r[start:stop]

#fit the power function to get rid off the signal decay and correct the profile
popt, pcov = curve_fit(func, r, prof, maxfev = 20000)
ctf_corrected = prof-func(r,*popt)
#ctf_corrected = ctf_corrected/np.max(ctf_corrected)

#fit the CTF function to corrected profile
p0 = [1,-30000,0.5]
params, pcov_ctf = curve_fit(ctf_function, r_scale(r),ctf_corrected,p0=p0,maxfev = 2000)
print(params)

#CTF fit visualisation
rad = lambda x,y: np.sqrt(x**2+y**2)
X,Y = np.meshgrid(X,Y)
image = util.invert(np.rot90(np.rot90(ctf_function(rad(X,Y),*params))))


#power = normalize(power)
#image = (image * np.max(power))
#power[0:int(image.shape[1]/2),0:int(image.shape[0]/2)] = image[0:int(image.shape[1]/2),0:int(image.shape[0]/2)]

plt.figure(figsize=(18,10))
plt.subplot(121)
plt.xlabel('Spatial frequency, 1/A')
plt.ylabel('Corrected signal')
plt.plot(r,prof-func(r,*popt))
plt.plot(r,ctf_function(r_scale(r),*params))
plt.subplot(122)
plt.imshow(power,cmap=plt.cm.gray)
plt.imshow(image[:,0:int(power.shape[0]/2)],cmap=plt.cm.gray)

plt.axis('off')
plt.autoscale('off')
plt.show()