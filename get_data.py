import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential

import matplotlib.image as mpimg  # mpimg 用于读取图片
import os  # 用于从文件夹读取，计数文件（注意不一定是图片）
import numpy as np
import scipy.misc
import scipy.signal
import scipy.ndimage
import cv2
import collections
import pandas as pd
from lead_in_csv import lead_in_csv
from preprocess import *
#实际图片读取路径 pic_train_orig_x
pic_train_orig_x = "data/5000_origin"
#实际csv读取路径 pic_train_orig_x
pic_train_orig_y = "data/over.csv"


def load_image(number=None,read=None):
    # 传如number限制时，一共就读numbeer个。
    # read为true不净化只切割，否则调用cut_delnoise_image.preprocess先切后净一步到位
    if number is not None and number != 'all':
        train_total_num = number
    # 否则，读取路径 pic_train_orig_x（= "data/del"）所有文件
    else:
        filelist = os.listdir(pic_train_orig_x)
        train_total_num = len(filelist)
        print(train_total_num)          # 打印读取对象数目 train_total_num
    train_set_x_orig_cache = []
    for pic_num in range(0, train_total_num):
        if read == True:
            #默认，循环读取jpg文件到 X_train_orig变量缓存
            X_train_orig = mpimg.imread(pic_train_orig_x + "/" + str(pic_num + 1) + '.jpg')
            #切割！！！！ 返回切割后list
            cut_im = cut_image(X_train_orig)
        else:
            X_train_orig = mpimg.imread(pic_train_orig_x + "/" + str(pic_num + 1) + '.jpg')
            cut_im = cut_delnoise_image(X_train_orig)
        for i in range(len(cut_im)):
            # train_set_x_orig_cache 缓存所有切割后的图片
            train_set_x_orig_cache.append(cut_im[i])
    # train_set_x_是array化后切割图像矩阵缓存
    train_set_x_ = np.array(train_set_x_orig_cache)
    # y train 读取,调用lead_in_csv文件的lead_in_csv函数
    train_set_y_ = np.array(lead_in_csv())
    #  因为每张图x切割为四，y也返回一个个字母，故总计算对象数乘四
    train_total_num = train_total_num * 4
    # 分割train/test取后0.8为训练集，前0.2算准确率当测试集
    train_set_x_orig = train_set_x_[int(0.2* train_total_num): train_total_num]
    test_set_x_orig = train_set_x_[0:int(0.2* train_total_num)]
    train_set_y_orig = train_set_y_[int(0.2* train_total_num): train_total_num]
    test_set_y_orig = train_set_y_[0:int(0.2* train_total_num)]
    # 将y元素转换为数字
    y_train_num_list = y_to_pure_num(y_to_list(train_set_y_orig))
    y_test_num_list = y_to_pure_num(y_to_list(test_set_y_orig))
    print(y_train_num_list)
    print(y_test_num_list)
    return train_set_x_orig, y_train_num_list, test_set_x_orig, y_test_num_list


def load_datasets():
    x, y, x_test, y_test = load_image(read=True)
    # x,y 成对绑定
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    # 小组打乱
    train_db = train_db.shuffle(1000).map(preprocess).batch(32)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess).batch(32)
    sample = next(iter(train_db))
    # 打印一个batch中x，y形状
    print(sample[0].shape, sample[1].shape)

    return train_db, test_db


#测试用
if __name__ == '__main__':
    x, y, x_test, y_test = load_image(read=True)

# =============================================================================
#
# train_set_x_orig, y_train_num_list, test_set_x_orig, y_test_num_list=load_dataset()
# for i in range(len(train_set_x_orig)):
#     plt.imshow(train_set_x_orig[i])
#     plt.title(y_train_num_list[i])
#     plt.show()
# =============================================================================

