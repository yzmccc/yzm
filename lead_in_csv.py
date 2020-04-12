# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 00:01:52 2019

@author: 南谷
"""

from csv import reader
import numpy as np
import os
# 读取 y 文件路径
y_path= r'data/over.csv'

def lead_in_csv():
    pic_train_orig_y = y_path
    label = []
    with open(pic_train_orig_y, 'rt', encoding='UTF-8') as raw_data:
        readers = reader(raw_data, delimiter=',')
    # x为读取的每行参数
        x = list(readers)
        data = np.array(x)
        print("csv 共计读取"+str(len(x))+"行")
    # csv中所有行注入list
        for i in range(len(x)):
            if len(data[i][0])!=4:
                print("position in line:"+str(i+1)+"as"+str(data[i]))
        for i in range(len(x)):
            s = data[i][0]
            for j in range(0, 4):
                label.append(s[j])
    # 返回一个个字母的list【'a','4'....】
    return label
