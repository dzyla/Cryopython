import cryopython as cp
import cv2
import numpy as np
import sys
import os
import easygui
import mrcfile


def show_image(image, scale=1,adjust=True):
    with mrcfile.open(image, permissive=True) as mrc:


        if len(mrc.data.shape) == 2:
            height, width = mrc.data.shape[:2]
            img = mrc.data
        else:
            height, width = mrc.data.shape[1:]
            img = mrc.data[0]
        if height > 1000:
            r = 1000 / img.shape[1]
            dim = (1000, int(img.shape[0] * r))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        elif scale != 1:
            r = img.shape[1] *scale
            dim = (int(r), int(img.shape[0] * scale))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if adjust:
            img = cp.adjust_hist(img, 1.2, 1.1, False)
        img = img - np.min(img)
        cv2.imshow(image, img)
        n = True
        check = True
        while n:
            k = cv2.waitKey(0)
            if k == 32:
                cv2.destroyAllWindows()
                print('Image accepted: '+image)
                n = False
                return image
            elif k == ord('n'):
                cv2.destroyAllWindows()
                print('rejected '+image)
                n = False
                return None
            elif k == 27:
                n = False
                cv2.destroyAllWindows()
                quit(1)
            else:
                print('Not understood - press space or n')


def load_folder():
    dir = easygui.diropenbox(msg='Choose a directory: ')
    file_list = os.listdir(dir)
    mrc_list = []
    for file in file_list:
        if '.mrc' in file or '.MRC' in file:
            if 'frames' not in file:
                mrc_list.append(file)
            elif 'frames' in file:
                print('Only frames here - I was expecting single frame MRC files.')
                input('This is it. OK?')
                quit(1)
    return dir,mrc_list



'''Main Program starts here'''

#check for the folder if exists and if command line settings are provided
# if len(sys.argv) == 2:
#     if os.path.isdir(sys.argv[1]):
#         dir = sys.argv[1]
#     else:
#         'Folder does not exist, try another one'
# elif len(sys.argv) == 1:
#     dir = input('Which folder to check? (Default Micrographs/) : ')
#     if dir == None or dir == '':
#         dir = 'Micrographs/'
#         if not os.path.isdir(dir):
#             print('\n\nFolder does not exist. Exiting...')
#             quit(0)

#here good images names will be kept
good_images = []

#list creation and checking the total number of files
dir, files_list = load_folder()
no_of_files = len(files_list)

#check if single or Movies
msg ="Micrographs are frames or single MRC micrographs?"
title = "Movie or single MRC?"
choices = ["Single MRC","MRC movie"]
reply = easygui.buttonbox(msg, choices=choices)


#the loop over file list
for n,file in enumerate(files_list):
    print('\n'+str(n)+ ' out of '+str(no_of_files)+' images.')
    #name = show_image(dir+'\\'+file)
    name = show_image(dir+'\\'+file)
    if name != None:
        name = name.replace(dir+'\\','') #remove the folder name
        if reply == 'MRC movie':
            name = name.replace('.mrc','_frames.mrc') #select only frames
        good_images.append(name)

    #save temporaty selection just in case something went wrong
    with open('selected_micrographs.temp', 'w') as fh_temp:
        fh_temp.write("\n".join(good_images))


#save selected files
with open('selected_micrographs.txt','w') as fh:
    print('\n\nSelected images written to selected_micrographs.txt')
    fh.write("\n".join(good_images))

#remove the temp selection
if os.path.isfile('selected_micrographs.temp'):
    os.remove('selected_micrographs.temp')

