import cryopython as cp
import matplotlib.pyplot as plt
import numpy as np

# do_all = False
#
# if do_all:
#     classes, selected_classes = cp.select_multiple_class(cp.open_image('run_it025_classes.mrcs', True))
#
#     for n,twoDclass in enumerate(classes):
#         file_list = ['Falcon_2012_06_12-17_02_43_0.mrc','Falcon_2012_06_12-17_14_17_0.mrc','Falcon_2012_06_12-17_17_05_0.mrc']
#         for file in file_list:
#             filename = file
#             search_image = cp.open_image(filename,False, True)
#
#
#             file_xy = file.replace('.mrc','')+'-'+selected_classes[n]+'.xy'
#             temp_size = twoDclass.shape[0]
#             x, y = cp.autopick(twoDclass,search_image,0.01,temp_size,file_xy,1)
#             cp.extract(filename.replace('.mrc','-'+selected_classes[n]),search_image,x,y,twoDclass)
# elif not do_all:
#     cp.average_classes_mrcs('temp/Falcon_2012_06_12-17_17_05_0-44.mrcs', False)


#cp.average3d_to_2d_cv2()

#cp.average_classes_mrcs('FoilHole_24160656_Data_24154994_24154995_20180424_1756.mrcs')

cp.check_mrcs_cv()