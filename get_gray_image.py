import os
import matplotlib.image as mpimg  #
import tensorflow as tf
import scipy.ndimage
import cv2

pic_train_orig_x = "data/train_delnoise"

save_path = "data/train_gray"


def load_dataset():
    filelist = os.listdir(pic_train_orig_x)
    train_total_num = len(filelist)
    for pic_num in range(0, train_total_num):
        X_train_orig = mpimg.imread(pic_train_orig_x + "/" + str(pic_num + 1) + '.jpg')

        x = scipy.ndimage.median_filter(X_train_orig, (2, 2, 3))
        x = cv2.fastNlMeansDenoising(x, h=10, templateWindowSize=7, searchWindowSize=21)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        print(x.shape)

        mpimg.imsave(save_path + "/" + str(pic_num + 1) + '.jpg', x)


load_dataset()
