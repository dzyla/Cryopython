import mrcfile
import cv2
import easygui
import numpy as np

def open_mrcs(mrcs_file):
    with mrcfile.open(mrcs_file) as mrc:
        if mrc.is_image_stack():
            return mrc.data
        elif len(mrc.data.shape) == 3:
            return mrc.data

def check_mrcs_cv():
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

if __name__ == "__main__":
    check_mrcs_cv()