def average3d_to_2d_cv2():
    import cv2
    import easygui
    import mrcfile
    import numpy as np
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