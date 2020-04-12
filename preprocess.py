import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import csv

import matplotlib.image as mpimg  # mpimg 用于读取图片
import os  # 用于从文件夹读取，计数文件（注意不一定是图片）
import numpy as np
import scipy.misc
import scipy.signal
import scipy.ndimage
import cv2
# import PIL as plt
def turn_to_gray(pic_list):
    total_num = len(pic_list)
    grayed_pic_list = []
    for i in range(0,total_num):
        pic_now = pic_list[i]
        gray_image = np.dot(pic_now[..., :3], [0.2989, 0.5870, 0.1140])
        grayed_pic_list.append(gray_image)
    return grayed_pic_list

def turn_to_gray(pic_list):
    total_num = len(pic_list)
    grayed_pic_list = []
    for i in range(0,total_num):
        pic_now = pic_list[i]
        gray_image = np.dot(pic_now[..., :3], [0.2989, 0.5870, 0.1140])
        grayed_pic_list.append(gray_image)
    return grayed_pic_list

def cut_delnoise_image(im):
    # 对单个图像切割并去噪

    im_cut_1 = im[0:40, 10:35]
    im_cut_2 = im[0:40, 35:60]
    im_cut_3 = im[0:40, 60:85]
    im_cut_4 = im[0:40, 85:110]

    im_cut = [im_cut_1, im_cut_2, im_cut_3, im_cut_4]
    print("preprocessing pics，may take you a long time")
    for i in range(4):
        #   只用del10净化，
        im_cut[i] = cv2.fastNlMeansDenoising(im_cut[i], h=10, templateWindowSize=7, searchWindowSize=21)
    return im_cut


def cut_image(im):
    # 对单个图像切割并去噪
    #im_cut_1 = tf.expand_dims(im[0:40, 10:35],-1)
    #im_cut_2 = tf.expand_dims(im[0:40, 35:60],-1)
    #im_cut_3 = tf.expand_dims(im[0:40, 60:85],-1)
   # im_cut_4 = tf.expand_dims(im[0:40, 85:110],-1)

    im_cut_1 = im[0:40, 10:35]
    im_cut_2 = im[0:40, 35:60]
    im_cut_3 = im[0:40, 60:85]
    im_cut_4 = im[0:40, 85:110]

    im_cut = [im_cut_1, im_cut_2, im_cut_3, im_cut_4]

    return im_cut

def y_to_list(y_array):
    y_list = y_array.tolist()
    return y_list


# 将数字字母转为62个数字
def y_to_pure_num(y_solid_alphabet):
    finall_num_y_pack = []
    for i in y_solid_alphabet:
        if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            finall_num_y_pack.append(int(i))
        elif i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                      't', 'u', 'v', 'w', 'x', 'y', 'z']:
            finall_num_y_pack.append(int(ord(i) - 87))  # a是ascii97号,10到35
        elif i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
            finall_num_y_pack.append(int(ord(i) - 29))  # A是ascii65号，36到61号
    return finall_num_y_pack

def preprocess(x, y):
    # [0-1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y



# 数值转换为4字字符串
def tensor_32_to_8_list(result):
    arr_of_tensor = np.array(result)
    to_char_temp = []
    last = []
    counter = 0
    for i in arr_of_tensor:
        # print("counter = " + str(counter))
        # print(to_char_temp)
        if i < 10:
            to_char_temp.append(str(int(i)))
        elif i < 36:
            to_char_temp.append((str(chr(int(i) + 87))))
        elif i < 62:
            to_char_temp.append(str(chr(int(i) + 29)))
        else:
            counter = counter - 1
        # print(to_char_temp)
        counter = counter + 1
        if counter == 4:
            counter = 0
            # print(to_char_temp)
            temp_str = to_char_temp[0] + to_char_temp[1] + to_char_temp[2] + to_char_temp[3]
            last.append(temp_str)
            to_char_temp = []
            # print("last now:" + str(last))
    return last

