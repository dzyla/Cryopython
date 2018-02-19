import cryopython as cp
import matplotlib.pyplot as plt
import numpy as np
#
# #image which is used for autopicking open_image(file, classes=False, movie=False):
search_image = cp.open_image('170715_110001.mrc',False, True)


# #Show calibrated histogram MRC file #adjust_hist(mrc_data,gamma,show_hist)
#cp.adjust_hist(search_image,1,True)


#choose 2D classes | select_class(mrcs_input,show_plot):
#twoDclass = cp.select_class(cp.open_image('run_ct1_it025_classes.mrcs', True),True)

factor = 1
#temp_size = twoDclass.shape[0]*factor

#autopick autopick(template,image,image_cal,threshold,size)

#x, y = cp.autopick(twoDclass,search_image,0.2, temp_size,1)

#extract particles | extract(image,coord_x, coord_y, template,size=0)
#cp.extract(search_image,x,y,twoDclass)

# import os
# for file in os.listdir("temp"):
#     #print(file)
#     if file.endswith(".mrc"):
#         cp.show_mrc('temp/'+file)


#do_powerspectrum(image, gamma, p1, p2, gaus_denoise)
cp.do_powerspectrum(search_image,5,0.1,99.9,3)