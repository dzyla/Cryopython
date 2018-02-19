from time import time
import os
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy import fftpack
from skimage import exposure
from skimage import util
from skimage.feature import match_template
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, color
from skimage.util import random_noise



'''-------old import---------'''


def open_mrc(mrc_file):
    with mrcfile.open(mrc_file) as mrc:
        if mrc.is_single_image():
            return mrc.data


def open_mrcs(mrcs_file):
    with mrcfile.open(mrcs_file) as mrc:
        if mrc.is_image_stack():
            return mrc.data


'''-------old import---------'''


def open_image(file, classes=False, movie=False):
    with mrcfile.open(file, permissive=True) as mrc:
        if mrc.is_single_image():
            print('This is single MRC picture')
            return mrc.data
        elif mrc.is_image_stack():
            if not classes:
                if not movie:
                    dimentions = mrc.data.shape
                    slice = input('\nThis is image stack of ' + str(dimentions[0]) + ' slices. Which one to choose: ')
                elif movie:
                    slice = 3
                return mrc.data[int(slice)]
            elif classes:
                return mrc.data
        elif mrc.is_volume_stack():
            print('File is volume stack. Nothing to do here')
            exit(0)
        else:
            print('Not recognized file. Exiting..')
            exit(0)


def adjust_hist(mrc_data, gamma, show_hist):
    # set values from 0 to 1
    # mrc_data = exposure.rescale_intensity(mrc_data, (0,1))

    # calc histogram for evaluation purposes
    hist_mean = (np.mean(mrc_data))
    hist_min, hist_max = (np.min(mrc_data), np.max(mrc_data))
    #hist = np.histogram(mrc_data.flatten(), 50)
    # print(hist_min,hist_max, hist_mean, hist)

    # contrast streaching for not correctly corrected images
    if hist_max > 5 * hist_mean:
        p2, p98 = np.percentile(mrc_data, (0.5, 98.0))  # change those values for better contrast
        mrc_data = exposure.rescale_intensity(mrc_data, in_range=(p2, p98))

    mrc_data = exposure.adjust_gamma(mrc_data, gamma)

    # logarithmic_corrected = exposure.equalize_hist(img_rescale)

    if show_hist:
        plt.figure(figsize=(10, 10))
        plt.imshow(mrc_data, cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()
    return mrc_data


def select_class(mrcs_input, show_plot):
    z, x, y = mrcs_input.shape
    image_row = int(np.sqrt(z)) + 1
    if show_plot:
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        plt.figure(figsize=(20, 18))
        for i in range(0, z):
            plt.subplot(image_row, image_row, i + 1)
            plt.axis('off')
            img1 = mrcs_input[i]
            plt.imshow(img1, cmap=plt.cm.gray)
            plt.title('Class: ' + str(i), size=5)
            plt.axis('off')
            plt.autoscale('off')
            # plt.tight_layout(pad=0.01, w_pad=0.5, h_pad=0.01)
        plt.show()
        class_select = int(input('Select class: '))
        print('Template size: ', mrcs_input[class_select].shape)
        return mrcs_input[class_select]
    else:
        print('Template size: ', mrcs_input[12].shape)
        return mrcs_input[0]


def do_powerspectrum(image, gamma, p1, p2, gaus_denoise):
    F1 = fftpack.fft2(image)
    F2 = fftpack.fftshift(F1)
    psd2D = np.abs(F2) ** 2
    # plt.subplot(111)

    # power spectrum
    # plt.imshow(np.log10(psd2D), cmap=plt.cm.gray)
    plt.axis('off')
    power_spec = np.log10(psd2D)

    p1, p2 = np.percentile(power_spec, (p1, p2))
    corrected = exposure.rescale_intensity(power_spec, in_range=(p1, p2))
    gamma_corrected = exposure.adjust_gamma(corrected, gamma)

    #denoize
    gauss_denoised = ndimage.gaussian_filter(gamma_corrected, gaus_denoise)
    img_shape = gauss_denoised.shape
    # For CTF correction
    # xy = np.unravel_index(power_spec.argmax(), power_spec.shape)
    # print('Center of the image: '+str(xy))
    #print(gamma_corrected[2048:2049+1,2048:])

    #plot
    plt.figure(figsize=(12,12))
    #plt.imshow(denoise_tv_chambolle(gamma_corrected, weight=0.15), cmap=plt.cm.gray)
    plt.imshow(gauss_denoised[int(img_shape[1]*0.25):int(img_shape[1]*0.75),int(img_shape[0]*0.25):int(img_shape[0]*0.75)],cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
    plt.tight_layout()
    # plt.subplot(122)

    # #angles
    # plt.imshow(np.angle(F2))
    plt.show()
    return gauss_denoised[int(img_shape[1]*0.25):int(img_shape[1]*0.75),int(img_shape[0]*0.25):int(img_shape[0]*0.75)]


def autopick(template, image, threshold, size, gamma=1):
    t_start = time()

    # reverse template image from negative to positive like micrograph
    template = util.invert(template)

    # do the cross-corelation between template and image
    result = match_template(image, template)

    # find coordinates of c-c map maxima

    neighborhood_size = size  # how many pixels should have the search window for a neighbour
    data_max = filters.maximum_filter(result, neighborhood_size)  # map of maxima
    maxima = (result == data_max)  # max value of dataset?

    data_min = filters.minimum_filter(result, neighborhood_size)  # map of minima
    diff = ((data_max - data_min) > threshold)  # if difference is greater than threshold
    maxima[diff == 0] = 0  #
    labeled, num_objects = ndimage.label(maxima)  # creates a map of found maxima and shows how many found

    temp_size = template.shape[0]  # get the shape of template

    print('\nFound ' + str(num_objects) + ' particles.')

    # generated slices from map of maxima
    slices = ndimage.find_objects(labeled)

    # define lists of coordinates
    x, y = [], []
    x_image, y_image = [], []
    x_temp, y_temp = [], []
    x_extract, y_extract = [], []
    match_value = []

    # calculate the coordinates required for analysis
    for dy, dx in slices:
        # cc quality
        match_value.append(data_max[dy, dx][0][0])
        # print(data_max[dy, dx][0][0])

        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

        # picture size is different and coordinates are moved by the size of template?
        x_image_center = (dx.start + dx.stop - 1 + temp_size) / 2
        y_image_center = (dy.start + dy.stop - 1 + temp_size) / 2

        # this one if for dots center in c-c map
        x_image.append(x_image_center)
        y_image.append(y_image_center)

        # this one is for rectangles in the image
        x_temp.append(x_image_center - 0.5 * temp_size)
        y_temp.append(y_image_center - 0.5 * temp_size)

        # this one is for the extract from image
        x_extract.append(x_image_center - 0.5 * temp_size)
        y_extract.append(y_image_center + 0.5 * temp_size)

    # zip both coordinates
    xy_coord = zip(x_temp, y_temp)

    # process cc data quality
    cc_min = min(match_value)
    cc_max = max(match_value)
    colors = cm.hot(np.arange(0, 101, 1))

    # define plots to show
    plt.figure(figsize=(40, 20))  # show reaaly big figure
    ax_image = plt.subplot(1, 2, 1)  # plot for the picture
    ax_diff = plt.subplot(1, 2, 2)  # plot for the c-c result

    # plot rectangles in the search image with found molecules
    for i, center in enumerate(xy_coord):
        ax_image.add_patch(
            patches.Rectangle(
                center,
                temp_size,
                temp_size,
                fill=False, ec=colors[int((match_value[i] - cc_min) / (cc_max - cc_min) * 100)]))  # remove background))

    ax_image.imshow(adjust_hist(image, gamma, False), cmap=plt.cm.gray)
    ax_image.set_axis_off()
    ax_image.autoscale(False)
    # ax_maxi.scatter(x_image, y_image, s=neighborhood_size, facecolors='none', edgecolors='r')

    ax_diff.imshow(result)
    ax_diff.scatter(x, y, s=neighborhood_size, facecolors='none', edgecolors='r')
    ax_diff.set_axis_off()
    ax_diff.autoscale(False)
    plt.show()
    print('Done in ' + str(round(time() - t_start, 2)) + ' s')

    '''---To do: template rotation---'''
    #     template_search = interpolation.rotate(template,i)
    #     result = match_template(search_image, template_search)
    #     plt.subplot(2,3,n)
    #     ij = np.unravel_index(np.argmax(result), result.shape)
    #     x, y = ij[::-1]
    #     plt.imshow(result)
    #     plt.autoscale(False)
    #     plt.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=50)
    #     n+=1
    '''------------------------------'''

    return x_extract, y_extract


def extract(image, coord_x, coord_y, template, size=0):
    extract = []

    # if size of the extracted area should be different than size of the template
    if size == 0:
        temp_size = template.shape[0]
    else:
        temp_size = int(size)

    # zip coords for extract
    xy_coord = zip(coord_x, coord_y)

    n = 0
    for num, coord in enumerate(xy_coord):
        x_start = int(coord[0])
        x_stop = int(coord[0] + temp_size)
        y_stop = int(coord[1])
        y_start = int(coord[1] - temp_size)

        # check if extracted area is still in the picture
        if x_start > 0 and x_stop > 0 and y_start > 0 and y_stop > 0:
            n += 1
            #extract.append(image[y_start:y_stop, x_start:x_stop, ])

            # save multiple mrc files
            save_mrc(num, image[y_start:y_stop, x_start:x_stop])
    print('Extracted ' + str(n) + ' particles in total')


def save_mrc(name, image):
    os.makedirs('temp', exist_ok=True)
    with mrcfile.new('temp/' + str(name) + '.mrc', overwrite=True) as mrc:
        mrc.set_data(image)


def show_mrc(file):
    with mrcfile.open(file) as mrc:
        if mrc.is_single_image():
            plt.imshow(mrc.data, cmap=plt.cm.gray)
            plt.show()


def show_mrcs(file, slice):
    with mrcfile.open(file) as mrc:
        if mrc.is_image_stack():
            plt.imshow(mrc.data[slice], cmap=plt.cm.gray)
            plt.axis('off')
            plt.title('Slice ' + str(slice))
            plt.show()

# def blur_the_template

# def rotate_the_template

# average_pictures

# show how good is the autopick with colors

# distance_picking
