import numpy as np
import matplotlib.pyplot as plt
import cryopython as cp
import matplotlib.cm as cm
from scipy import optimize
from scipy import optimize
from scipy.optimize import curve_fit

#open image
pixel_size = 1.7 #Angstrom

#do_powerspectrum(image, gamma, p1, p2, gaus_denoise)
image = cp.do_powerspectrum(cp.open_image('170715_140002.mrc'),1,0,100,1.2)
#image = image[5:,5:]
#print(image.shape)

# k = np.linspace(1/100,1/(2*pixel_size),image.shape[0])
Cs = 2
defocus = -40000

ht = 200   #kV
A = 0.15
sf = 1 #scaling factor
B = 150

#calculated constants
wavelength = 12.398 / np.sqrt(ht * (1022 + ht))  #for kV
#defocus_scherzer = -1.2*np.sqrt(Cs*wavelength)


#function description with envelope function
def ctf_function(k,sf,defocus,offset):
    B=250
    ctf = -2 * np.pi * (Cs * wavelength ** 3 * (k ** 4) / 4 - np.pi * defocus * wavelength * (k ** 2) / 2)
    CTF = -sf * ((1 - A ** 2) ** 0.5 * np.sin(ctf) + A * np.cos(ctf))
    env = np.exp(-B * k ** 2)
    return env*CTF+offset

#spatial frequency
k = np.linspace(1/100,1/(2*pixel_size),image.shape[0]/2)
X,Y = np.meshgrid(np.linspace(-1/(2*pixel_size),1/(2*pixel_size),image.shape[0]),np.linspace(-1/(2*pixel_size),1/(2*pixel_size),image.shape[0]))

#create array of radii
x,y = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
print(x.shape,y.shape)
x0, y0 = xy = np.unravel_index(image.argmax(), image.shape)
print('Center of the image: '+str(x0)+' '+str(y0))
R = np.sqrt((x-x0)**2+(y-y0)**2)

#prepare image to fit
space = 0.5
f = lambda r : np.mean(image[(R >= r-space) & (R < r+space)])
r  = np.linspace(0,image.shape[0],num=image.shape[0])
mean = np.vectorize(f)(r)
mean = mean[:int(image.shape[0]/2)]

with open('mean.txt', 'a') as fh:
    for i in mean:
        print(i,file=fh)

print('done creating mean')
end = 200
#mean = mean[0:end]
k_red = k

#fitting #sf,B,defocus,offset
p0 = [1,-30000,0.5]
params, pcov = curve_fit(ctf_function, k_red,mean,p0=p0,maxfev = 2000)
print(params)

#changing to radial coordinates
rad = lambda x,y: np.sqrt(x**2+y**2)
image = ctf_function(rad(X,Y),*params)

#plot setup
fig, (ax,ax2) = plt.subplots(ncols=2)
ax.plot(k_red, ctf_function(k_red,*params))
ax.plot(k_red[:300],mean[:300])
ax2.imshow(image, extent=[X.min(),X.max(),Y.min(),Y.max()], cmap=plt.cm.gray)
plt.show()




#proper functions
# ctf = -2*np.pi*(Cs * wavelength**3 * (k**4)/4 - np.pi * defocus * wavelength * (k**2)/2)
# CTF = -sf * ((1-A**2)**0.5*np.sin(ctf) + A*np.cos(ctf))

#guassian envelope funtion
#env = np.exp(-B*k**2)