import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
import matplotlib.image as mpimg  # mpimg 用于读取图片 x
import pandas as pd  # 导入 y train
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import os  # 用于从文件夹读取，计数文件（注意不一定是图片）
import numpy as np
import scipy.misc
import scipy.signal
import scipy.ndimage
import cv2


pic_train_orig_x = "data/train"
pic_train_orig_y = "data/train_label.csv"

pic_wanna_orig = "data/test"
# 独热eye元素个数
eye_num = 62


# y_array -> y_list
def y_to_list(y_array):
    y_list = y_array.tolist()
    finall_box = []
    mini_box = []
    for i in y_list:
        mini_box.append(i[0][0])
        mini_box.append(i[0][1])
        mini_box.append(i[0][2])
        mini_box.append(i[0][3])
        # print(mini_box)
        finall_box.append(mini_box)
        mini_box = []
    return finall_box


# 输出如 ：[['4', 'J', 'y', '3'], ['P', '6', 'B', 'f'], ['v', 'J', 'l', 'D'].......]]

# 将数字字母转为62个数字
def y_to_pure_num(y_solid_alphabet):
    finall_num_y_pack = []
    for i in y_solid_alphabet:
        cache = []
        for m in range(0, 4):
            if i[m] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                cache.append(int(i[m]))
            elif i[m] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                          't', 'u', 'v', 'w', 'x', 'y', 'z']:
                cache.append(int(ord(i[m]) - 87))  # a是ascii97号,10到35
            elif i[m] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                          'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
                cache.append(int(ord(i[m]) - 29))  # A是ascii65号，36到61号
        finall_num_y_pack.append(cache)
    return finall_num_y_pack


# 输出如 ：[[4, 44, 34, 3], .......]]


def load_dataset():
    filelist = os.listdir(pic_train_orig_x)
    train_total_num = len(filelist)
    train_set_x_orig_cache = []
    for pic_num in range(0, train_total_num):
        X_train_orig = mpimg.imread(pic_train_orig_x + "/" + str(pic_num + 1) + '.jpg')
        train_set_x_orig_cache.append(X_train_orig)
    # x train 读取
    train_set_x_ = np.array(train_set_x_orig_cache)
    # y train 读取
    train_set_y_orig_cache = pd.read_csv(pic_train_orig_y, header=None)
    train_set_y_ = np.array(train_set_y_orig_cache)
    # print(train_set_x_)

    # train_set_x_  train_set_y_
    # train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig,
    #  分割为 6 : 4
    train_set_x_orig = train_set_x_[:int(0.6 * train_total_num)]
    test_set_x_orig = train_set_x_[int(0.6 * train_total_num): train_total_num]
    train_set_y_orig = train_set_y_[:int(0.6 * train_total_num)]
    test_set_y_orig = train_set_y_[int(0.6 * train_total_num): train_total_num]

    y_train_num_list = y_to_pure_num(y_to_list(train_set_y_orig))
    y_test_num_list = y_to_pure_num(y_to_list(test_set_y_orig))
    print(y_train_num_list)
    return train_set_x_orig, y_train_num_list, test_set_x_orig, y_test_num_list


def preprocess(x, y):
    # [0-1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def convert_to_one_hot2(Y, C=62):
    print("转换为独热")
    temp=np.array(Y)
    ##################### 对Y长度统一性进行检查
    print("开始对Y长度统一性进行检查")
    for i in range(0,len(Y)):
        if len(Y[i]) != 4:
            print("第"+ str(i+1) +"个错误")
            print(Y[i])
    print("this time ,Y-len:"+str(len(Y)))
    print("结束对Y长度统一性进行检查")
    #####################
    Y = np.eye(C)[temp.reshape(len(Y),4)]
    print("转换结束")
    return Y


def batch_set():

    train_db, y, test_db, y_test = load_dataset()
    y_onehot = convert_to_one_hot2(y)
    print(y_onehot)
    train_db = tf.convert_to_tensor(train_db,dtype=tf.int32)
    y = tf.convert_to_tensor(y,dtype=tf.int32)
    test_db=tf.convert_to_tensor(test_db,dtype=tf.int32)
    y_test=tf.convert_to_tensor(y_test,dtype=tf.int32)

    print(train_db.shape, y.shape)

    train_db = train_db.shuffle(1000).map(preprocess).batch(64)

    test_db = test_db.map(preprocess).batch(64)
    sample = next(iter(train_db))
    print(sample[0].shape)
    return train_db,y,test_db,y_test
if __name__ == '__main__':
    batch_set()
