import datetime
import os
from time import time
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

'''-------old import---------'''


def open_mrc(mrc_file):
    with mrcfile.open(mrc_file) as mrc:
        if mrc.is_single_image():
            return mrc.data


def open_mrcs(mrcs_file):
    with mrcfile.open(mrcs_file) as mrc:
        if mrc.is_image_stack():
            return mrc.data
        elif len(mrc.data.shape) == 3:
            return mrc.data


'''-------old import---------'''


def open_image(file, classes=False, movie=False):
    with mrcfile.open(file, permissive=True) as mrc:
        if mrc.is_single_image():
            # print('This is single MRC picture')
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
        elif len(mrc.data.shape) == 3 and not mrc.is_image_stack():
            dimentions = mrc.data.shape
            slice = input('\nThis is image stack of ' + str(dimentions[0]) + ' slices. Which one to choose: ')
            return mrc.data[int(slice)]
        else:
            print('Not recognized file. Exiting..')
            exit(0)


def adjust_hist(mrc_data, gamma, denoise, show_hist, hist_corr=True):
    # set values from 0 to 1
    # mrc_data = exposure.rescale_intensity(mrc_data, (0,1))

    # contrast streaching for not correctly corrected images
    if hist_corr:
        # calc histogram for evaluation purposes
        hist_mean = (np.mean(mrc_data))
        hist_min, hist_max = (np.min(mrc_data), np.max(mrc_data))
        # hist = np.histogram(mrc_data.flatten(), 50)
        # print(hist_min,hist_max, hist_mean, hist)
        p1, p2 = np.percentile(mrc_data, (0.2, 98.0))  # change those values for better contrast
        mrc_data = exposure.rescale_intensity(mrc_data, in_range=(p1, p2))
        mrc_data = exposure.adjust_gamma(mrc_data, gamma)

    if denoise != 0:
        mrc_data = ndimage.gaussian_filter(mrc_data, denoise)
    # logarithmic_corrected = exposure.equalize_hist(img_rescale)

    if show_hist:
        plt.figure(figsize=(20, 20))
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
        print('Template size: ', mrcs_input[0].shape)
        return mrcs_input[0]

def select_multiple_class(mrcs_input):
    z, x, y = mrcs_input.shape
    image_row = int(np.sqrt(z)) + 1
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
    class_select = str((input('Select class spaced by ",": ')))
    class_select = class_select.split(',')
    n = 0
    if len(class_select) > 1:
        for number,c in enumerate(class_select):
            if n==0:
                classes = mrcs_input[int(c)]
                classes = np.expand_dims(classes,axis=0)
            else:
                classes = np.concatenate([classes, np.expand_dims(mrcs_input[int(c)],axis=0)], axis=0)
            n += 1
        print(str(n)+' classes chosen')
        return classes, class_select
    else:
        #print(mrcs_input[int(class_select[0])])
        classes = mrcs_input[0]
        classes = np.expand_dims(classes, axis=0)
        return classes, class_select


def average_classes(dir):
    file_list = (os.listdir(dir))
    for iteration, file in enumerate(file_list):
        if iteration == 0:
            shape = open_mrc(dir + file).shape
            average = np.zeros(shape)
        average += open_mrc(dir + file)
    average *= 1 / average.max()
    average -= np.mean(average)
    plt.imshow(average, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()
    return average


def average_classes_mrcs(path,multifile=False):
    if not multifile:
        print('Avering single stack!')
        with mrcfile.open(path) as mrc:
            for iteration, particle in enumerate(mrc.data):
                if iteration == 0:
                    shape = mrc.data.shape[1:]
                    average = np.zeros(shape)
                average += particle
            average -= np.mean(average)
            average *= 1 / average.max()
            average = util.invert(average)
            average = np.float32(average)
            plt.imshow(average, cmap=plt.cm.gray)
            plt.axis('off')
            plt.show()
            save_mrc('average.mrc',average)
            return average
    elif multifile:
        print('Averaging multiple stacks!')
        file_list = os.listdir(path)
        for file_num,file in enumerate(file_list):
            n = 0
            if '.mrcs' in file:
                with mrcfile.open(str(path)+str(file)) as mrc:
                    for iteration, particle in enumerate(mrc.data):
                        if iteration == 0 and n == 0:
                            shape = mrc.data.shape[1:]
                            average = np.zeros(shape)
                        average += particle
                        n +=1
        average -= np.mean(average)
        average *= 1 / average.max()
        average = util.invert(average)
        average = np.float32(average)
        plt.imshow(average, cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = 'average-'+str(date)
        save_mrc(name, average)
        return average


def select_class_multicore(mrcs_input):
    z, x, y = mrcs_input.shape
    image_row = int(np.sqrt(z)) + 1
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
    return class_select


def do_powerspectrum(image, gamma, p1, p2, gaus_denoise, averaging=False, plot=False):
    # image = exposure.rescale_intensity(image, (0.01,1))

    if averaging:
        if image.shape[0] != image.shape[1]:
            print('Image is not square! Exiting...')
            exit(1)
        im_sum = 1
        ave_fac = 0.5
        shape = image.shape[0]
        # average = np.zeros((int(ave_fac * shape), int(ave_fac * shape)))
        average = np.zeros((int(ave_fac * shape), int(ave_fac * shape)))
        # do the powerspectrum averaging for 4 parts of the image
        for x in range(1, int(1 / ave_fac + 1)):
            for y in range(1, int(1 / ave_fac + 1)):
                sub_image = image[int((y - 1) * ave_fac * shape):int(y * ave_fac * shape),
                            int((x - 1) * ave_fac * shape):int(x * ave_fac * shape)]
                F1 = fftpack.fft2(sub_image)
                F2 = fftpack.fftshift(F1)
                # psd2D = np.abs(F2) ** 2
                psd2D = np.abs(F2) ** 2
                if psd2D.shape == average.shape:
                    average += psd2D
                    im_sum += 1
        average = average / im_sum
        psd2D = average
    # image = exposure.rescale_intensity(image, (0.01,1))
    F1 = fftpack.fft2(image)
    F2 = fftpack.fftshift(F1)
    # psd2D = np.abs(F2) ** 2
    psd2D = np.abs(F2) ** 2

    # power spectrum
    power_spec = np.log10(psd2D)

    # power spectrum
    power_spec = np.log10(psd2D)
    # power_spec = exposure.rescale_intensity(power_spec, (0,1))

    p1, p2 = np.percentile(power_spec, (p1, p2))
    corrected = exposure.rescale_intensity(power_spec, in_range=(p1, p2))
    gamma_corrected = exposure.adjust_gamma(corrected, gamma)

    # denoize
    gauss_denoised = ndimage.gaussian_filter(gamma_corrected, gaus_denoise)
    img_shape = gauss_denoised.shape
    # For CTF correction
    # xy = np.unravel_index(power_spec.argmax(), power_spec.shape)
    # print('Center of the image: '+str(xy))
    # print(gamma_corrected[2048:2049+1,2048:])

    # plot
    if plot:
        plt.figure(figsize=(12, 12))
        # plt.imshow(gauss_denoised[int(img_shape[1]*0.25):int(img_shape[1]*0.75),int(img_shape[0]*0.25):int(img_shape[0]*0.75)],cmap=plt.cm.gray)
        plt.imshow(gauss_denoised, cmap=plt.cm.gray)
        plt.axis('off')
        plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
        plt.tight_layout()
        # plt.subplot(122)

        # #angles
        # plt.imshow(np.angle(F2))
        plt.show()
    # return gauss_denoised[int(img_shape[1]*0.25):int(img_shape[1]*0.75),int(img_shape[0]*0.25):int(img_shape[0]*0.75)]
    return gauss_denoised


def do_powersec_raw(image):
    # image = exposure.rescale_intensity(image, (0.01,1))
    F1 = fftpack.fft2(image)
    F2 = fftpack.fftshift(F1)
    # psd2D = np.abs(F2) ** 2
    psd2D = np.abs(F2) ** 2

    # power spectrum
    power_spec = np.log10(psd2D)
    return power_spec


def autopick(template, image, threshold, size, filename, denoise=1, gamma=1):
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

    #extract particles
    with open(filename, 'w') as fileout:
        for xy in xy_coord:
            fileout.write("\n".join(str(xy)))


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

    ax_image.imshow(adjust_hist(image, gamma, denoise, False), cmap=plt.cm.gray)
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


def extract(filename, image, coord_x, coord_y, template, size=0):
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
            # extract.append(image[y_start:y_stop, x_start:x_stop, ])

            # save multiple mrc files
            if num < 10:
                num = '00' + str(num)
            elif 9 < num < 100:
                num = '0' + str(num)
            else:
                num = str(num)
            # save_mrc(num, image[y_start:y_stop, x_start:x_stop])
            # save_mrcs_stack(filename, image[y_start:y_stop, x_start:x_stop])
            particle = np.expand_dims(image[y_start:y_stop, x_start:x_stop], axis=0)
            if n == 0:
                particle_list = particle
            elif n != 0:
                particle_list = np.concatenate([particle_list, particle], axis=0)
            n += 1

    with mrcfile.new('temp/' + filename.replace('.mrc', '') + '.mrcs', overwrite=True) as mrc:
        mrc.set_data(particle_list)
    print('Extracted ' + str(n) + ' particles in total')


def save_mrc(name, image):
    os.makedirs('temp', exist_ok=True)
    with mrcfile.new('temp/' + str(name) + '.mrc', overwrite=True) as mrc:
        mrc.set_data(image)


def save_mrcs_stack(name, image):
    import os.path
    name = name.replace('.mrc', '')
    if not os.path.isfile('temp/' + name + '.mrcs'):
        os.makedirs('temp', exist_ok=True)
        with mrcfile.new('temp/' + str(name) + '.mrcs', overwrite=True) as mrc:
            image = np.expand_dims(image, axis=0)
            mrc.set_data(image)
    else:
        # with mrcfile.open('temp/'+name+'.mrcs', mode='r+') as mrc:
        with open('temp/' + name + '.mrcs', mode='a') as mrc:
            image = np.expand_dims(image, axis=0)
            new = np.concatenate([mrc.data, image], axis=0)
            mrc.set_data(new)


def show_mrc(file):
    with mrcfile.open(file) as mrc:
        if mrc.is_single_image():
            plt.imshow(mrc.data, cmap=plt.cm.gray)
            plt.show()

def normalize_particle(particle):
    particle = (particle - np.min(particle)) / (np.max(particle) - np.min(particle))
    particle = util.invert(particle)
    return particle

def show_image(img,scale=1,adjust=None):
    import cv2
    if adjust == None:
        img -= np.min(img)
        img = adjust_hist(img, 1.2, 1.1, False)
    elif adjust != None:
        img -= np.min(img)
        img = adjust_hist(img, adjust, 1.1, False)
    height, width = img.shape[:2]
    if height > 1000:
        r = 1000 / img.shape[1]
        dim = (1000, int(img.shape[0] * r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    elif scale != 1:
        r = img.shape[1] * scale
        dim = (int(r), int(img.shape[0] * scale))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_mrcs(file, slice):
    with mrcfile.open(file) as mrc:
        if mrc.is_image_stack():
            plt.imshow(mrc.data[slice], cmap=plt.cm.gray)
            plt.axis('off')
            plt.title('Slice ' + str(slice))
            plt.show()


def slice_xy(image_mrcs, plane):
    z, x, y = image_mrcs.shape
    if x != y and x != z:
        print('It is not a cube volume!')
    else:
        image_row = 30
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        plt.figure(figsize=(20, 18))
        for i in range(0, x):
            plt.subplot(image_row, image_row, i + 1)
            plt.axis('off')
            img1 = image_mrcs[:, i, :]
            plt.imshow(img1, cmap=plt.cm.gray)
            plt.title('Class: ' + str(i), size=5)
            plt.axis('off')
            plt.autoscale('off')
            # plt.tight_layout(pad=0.01, w_pad=0.5, h_pad=0.01)
        plt.show()
        class_select = int(input('Select class: '))
        print('Template size: ', image_mrcs[class_select].shape)
        return image_mrcs[class_select]


def average3d_to_2d(mrcs_file, axis, limit_low, limit_high):

    with mrcfile.open(mrcs_file) as volume:

        mrcs_file = volume.data
        z, x, y =  volume.data.shape
        if limit_low == limit_high:
            limit_low = 0
            limit_high = x

        map = plt.cm.bone

        if axis == 1:
            average = np.zeros((z, y))
            for i in range(limit_low, limit_high):
                img = mrcs_file[:, i, :]
                average += img
            average = (average - np.min(average)) / (np.max(average) - np.min(average))
            average = adjust_hist(average, 1.2, 1, False)
            plt.imshow(average, cmap=map)
            plt.axis('off')
            plt.autoscale('off')
            plt.show()

        elif axis == 0:
            average = np.zeros((z, y))
            for i in range(limit_low, limit_high):
                img = mrcs_file[i, :, :]
                average += img
            average = (average - np.min(average)) / (np.max(average) - np.min(average))
            average = adjust_hist(average,1.2,1,False)
            plt.imshow(average, cmap=map)
            plt.axis('off')
            plt.autoscale('off')
            plt.show()
        elif axis == 2:
            average = np.zeros((z, y))
            for i in range(limit_low, limit_high):
                img = mrcs_file[:, :, i]
                average += img
            average = (average - np.min(average)) / (np.max(average) - np.min(average))
            average = adjust_hist(average, 1.2, 1, False)
            plt.imshow(average, cmap=map)
            plt.axis('off')
            plt.autoscale('off')
            plt.show()
        return average

def average3d_to_2d_cv2():
    import cv2
    import easygui
    mrcs_file = easygui.fileopenbox()


    def nothing(x):
        pass

    with mrcfile.open(mrcs_file) as volume:

        mrcs_file = volume.data
        z, x, y = volume.data.shape

        cv2.namedWindow('Volume')
        cv2.namedWindow('Volume', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Volume', 500, 500)
        cv2.createTrackbar('Axis', 'Volume', 0, 2, nothing)
        cv2.createTrackbar('Lower limit', 'Volume', 0, x, nothing)
        cv2.createTrackbar('Higer limit', 'Volume', 0, x, nothing)
        average = np.zeros((z, y))
        while (1):
            axis = cv2.getTrackbarPos('Axis', 'Volume')
            limit_low = cv2.getTrackbarPos('Lower limit', 'Volume')
            limit_high = cv2.getTrackbarPos('Higer limit', 'Volume')

            if axis == 1:
                average = np.zeros((z, y))
                for i in range(limit_low, limit_high):
                    img = mrcs_file[:, i, :]
                    average += img
                average = (average - np.min(average)) / (np.max(average) - np.min(average))
                average = cv2.resize(average, (500, 500))
                #average = adjust_hist(average, 1.2, 1, False)
            elif axis == 0:
                average = np.zeros((z, y))
                for i in range(limit_low, limit_high):
                    img = mrcs_file[i, :, :]
                    average += img
                average = (average - np.min(average)) / (np.max(average) - np.min(average))
                average = cv2.resize(average, (500, 500))

                #average = adjust_hist(average,1.2,1,False)
            elif axis == 2:
                average = np.zeros((z, y))
                for i in range(limit_low, limit_high):
                    img = mrcs_file[:, :, i]
                    average += img
                average = (average - np.min(average)) / (np.max(average) - np.min(average))
                average = cv2.resize(average, (500, 500))

                #average = adjust_hist(average, 1.2, 1, False)

            cv2.imshow('Volume', average)
            cv2.namedWindow('Volume', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Volume', 500, 500)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                quit(1)

def check_mrcs_cv():
    import cv2
    import easygui
    mrcs_file = easygui.fileopenbox()

    def nothing(x):
        pass

    with mrcfile.open(mrcs_file) as mrc_stack:

        mrcs_file = mrc_stack.data
        z, x, y = mrc_stack.data.shape

        cv2.namedWindow('ImageStack')
        cv2.namedWindow('ImageStack', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('ImageStack', 500, 500)
        cv2.createTrackbar('Particle', 'ImageStack', 0, z, nothing)
        while (1):
            slice = cv2.getTrackbarPos('Particle', 'ImageStack')

            average = np.zeros((z, y))
            img = mrcs_file[slice-1, :, :]
            average = img
            if (np.max(average) - np.min(average)) != 0:
                average = (average - np.min(average)) / (np.max(average) - np.min(average))
            average = cv2.resize(average, (500, 500))

            cv2.imshow('ImageStack', average)
            cv2.namedWindow('ImageStack', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ImageStack', 500, 500)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                quit(1)

# def blur_the_template

# def rotate_the_template

# average_pictures

# distance_picking
