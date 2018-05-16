import os
import easygui
import cryopython as cp
import scipy as sp
import scipy.misc
import numpy as np
import imreg_dft as ird #does the translation and rotational search in FFT
from scipy import ndimage
from scipy.ndimage import fourier_shift
from skimage import util
import matplotlib.pyplot as plt
import cv2

def createCircularMask(h, w, radius=None):

    center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def mask_particle(particle,radius=None):
    h, w = particle.shape[:2]
    mask = createCircularMask(h, w,radius)
    masked_img = particle.copy()
    masked_img[~mask] = 0
    return masked_img

def mask_particle_center(particle,radius=None):
    h, w = particle.shape[:2]
    mask = createCircularMask(h, w,radius)
    masked_img = particle.copy()
    masked_img[mask] = 0
    return masked_img


def normalize_particle(particle):
    particle = (particle - np.min(particle)) / (np.max(particle) - np.min(particle))
    particle = util.invert(particle)
    return particle

def load_folder():
    dir = easygui.diropenbox(msg='Choose a directory: ')
    file_list = os.listdir(dir)
    mrcs_list = []
    for file in file_list:
        if '.mrcs' in file or '.MRCs' in file:
            mrcs_list.append(file)
    return dir,mrcs_list

def nothing(x):
    pass

dir, files_list = load_folder()
no_of_files = len(files_list)


msg ="Do masking or just check the size of mask?"
title = "Do masking or just check the size of mask?"
choices = ["Mask particles","Check the mask"]
reply = easygui.buttonbox(msg, choices=choices)

if reply == "Mask particles":
    radius = easygui.integerbox('Provide the mask radius','Mask radius')

    for n,file in enumerate(files_list):
        #print('\n'+str(n)+ ' out of '+str(no_of_files)+' images.')
        name = str(dir+'\\'+file)
        mrcs_stack = cp.open_mrcs(name)
        for particle in mrcs_stack:
            particle_masked = mask_particle_center(particle, radius)
            #cp.show_image(particle_masked)

            #add here array creation and saving masked stack!


elif reply == 'Check the mask':

    name = str(dir + '\\' + str(files_list[0]))
    mrcs_file = cp.open_mrcs(name)

    z, x, y = mrcs_file.shape

    img = mrcs_file[0]
    h, w = img.shape[:2]

    cv2.namedWindow('Particle')
    cv2.namedWindow('Particle', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('ImageStack', 500, 500)
    cv2.createTrackbar('Radius', 'Particle', 0, int(h/2), nothing)

    while (1):
        radius = cv2.getTrackbarPos('Radius', 'Particle')
        average = mask_particle_center(img, radius)
        average = cv2.resize(average, (500, 500))

        cv2.imshow('Particle', average)
        cv2.namedWindow('Particle', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Particle', 500, 500)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            quit(1)
