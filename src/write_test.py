import numpy as np
import struct
import scipy.io as sio
import numpy as np
import scipy.io as io
import math, os, os.path
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import time
from numpy.linalg import lstsq


# 读取txt文件数据
def readfile(filename):
    data_list = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            if lines != '\n':
                temp = round(float(lines), 4)
                data_list.append(temp)  # 添加新读取的数据
    data_list = np.array(data_list,dtype='float')  # 将数据从list类型转换为array类型。
    return data_list


def write_x():
    """
    编写4号树木的x轴输入信息
    """
    ospath = '../Data2/实验室4号树木/'
    ospathnames = [name for name in os.listdir(ospath)
                   if os.path.isdir(os.path.join(ospath, name))]
    for i in range(len(ospathnames)):
        ospathnames[i] = ospath + ospathnames[i] + '/'
    outcome = []
    for osname in ospathnames:
        filenames = [name for name in os.listdir(osname)
                     if os.path.isfile(os.path.join(osname, name)) and
                     name.endswith('树莓派.txt')]
        for i in range(len(filenames)):
            filenames[i] = osname + filenames[i]
        locat_filename = osname + 'location.txt'
        locat_list = [[] for i in range(8)]
        with open(locat_filename, 'r', encoding='utf-8') as file_to_read:
            for i in range(8):
                lines = file_to_read.readline()  # 整行读取数据
                nums = re.split(',|\n', lines)
                locat_list[i].append(nums[0])  # 添加新读取的数据
                locat_list[i].append(nums[1])  # 添加新读取的数据
        locat_list = np.array(locat_list).reshape(16)  # 将数据从list类型转换为array类型。

        for i in range(len(filenames)):
            data_list = readfile(filenames[i])
            data_list = np.concatenate((data_list, locat_list), axis=0)
            outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 44)
    np.save('../Data/label_Data/4号树木x.npy', outcome)

def write_y():
    """
    这个函数不会被调用，用于编写y轴信息
    """
    label=np.loadtxt('../Defect_Data/label4.txt',encoding='utf-8',dtype='int')
    label=np.array(label).reshape(-1)
    temp=np.zeros(shape=(5476,10000),dtype='int')
    temp[:,0:10000]=label
    pass

def write_test_x():
    """
    编写1-3号树木的输入x信息，秩序改写osname和np.save的文件名
    """
    osname = '../Data/实验室3号树木/'
    filenames = [name for name in os.listdir(osname)
                 if os.path.isfile(os.path.join(osname, name)) and
                 (name.startswith('树莓派') or
                  name.startswith('20') or
                  name.startswith('手按'))]
    for i in range(len(filenames)):
        filenames[i] = osname + filenames[i]
    locat_filename = osname + 'location.txt'
    locat_list = [[] for i in range(8)]
    with open(locat_filename, 'r', encoding='utf-8') as file_to_read:
        for i in range(8):
            lines = file_to_read.readline()  # 整行读取数据
            nums = re.split(',|\n', lines)
            locat_list[i].append(nums[0])  # 添加新读取的数据
            locat_list[i].append(nums[1])  # 添加新读取的数据
    locat_list = np.array(locat_list).reshape(16)  # 将数据从list类型转换为array类型。
    outcome=[]
    for i in range(len(filenames)):
        data_list = readfile(filenames[i])
        data_list = np.concatenate((data_list, locat_list), axis=0)
        outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 44)
    np.save('../Defect_Data/3号树木x.npy', outcome)
    pass

def write_only_time(osname):
    """
    编写1-3号树木的输入x信息，秩序改写osname和np.save的文件名
    """

    filenames = [name for name in os.listdir(osname)
                 if os.path.isfile(os.path.join(osname, name)) and
                 (name.endswith('树莓派.txt') or
                  name.startswith('20') or
                  name.startswith('手按'))]
    for i in range(len(filenames)):
        filenames[i] = osname + filenames[i]
    outcome=[]
    for i in range(len(filenames)):
        data_list = readfile(filenames[i])
        outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 28)
    np.save('../Data3/Data_npy/实验室2号树木/2号树木501x28.npy', outcome)

def find_wrong_x():
    """
    查找x轴输入信息
    """
    ospath = 'D:/02_soft_temp/py_temp/Raspberry-Communication/Data2/实验室4号树木/'
    ospathnames = [name for name in os.listdir(ospath)
                   if os.path.isdir(os.path.join(ospath, name))]
    for i in range(len(ospathnames)):
        ospathnames[i] = ospath + ospathnames[i] + '/'
    outcome = []
    for osname in ospathnames:
        filenames = [name for name in os.listdir(osname)
                     if os.path.isfile(os.path.join(osname, name)) and
                     name.endswith('树莓派.txt')]
        for i in range(len(filenames)):
            filenames[i] = osname + filenames[i]

        print(len(filenames))
        for i in range(len(filenames)):
            data_list = readfile(filenames[i])
            if data_list.shape[0]<28:
                print(filenames[i])
                try:
                    os.remove(filenames[i]) #删除数据错误文件
                except:
                    print("找不到:",filenames[i])
                continue
            for j in data_list:
                if j > 3:
                    print(filenames[i])
                    # try:
                    #     os.remove(filenames[i]) #删除数据错误文件
                    # except:
                    #     print("找不到:",filenames[i])
                    # continue
            outcome.append(data_list)

if __name__ == '__main__':
    write_only_time(osname = '../Data3/Data_file/实验室2号树木/')