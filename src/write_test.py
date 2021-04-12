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


NodeA_num=8
NodeB_num=8

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
    ospath = '../Data3/Data_npy/实验室1号树木/1号树木501x28.npy'
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

def readnpy_writex():
    """
    编写4号树木的x轴输入信息
    """
    filename = '../Data/label_Data/3号树木x.npy'
    locatename='../Data3/Data_npy/实验室3号树木/location.txt'


    outcome = []
    locat_list = [[] for i in range(NodeA_num)]
    with open(locatename, 'r', encoding='utf-8') as file_to_read:
        for i in range(8):
            lines = file_to_read.readline()  # 整行读取数据
            nums = re.split(',|\n', lines)
            locat_list[i].append(nums[0])  # 添加新读取的数据
            locat_list[i].append(nums[1])  # 添加新读取的数据
    locat_list = np.array(locat_list).reshape(NodeA_num*2)  # 将数据从list类型转换为array类型。
    data_list=np.load(filename)
    # locat_npy=np.zeros(shape=(data_list.shape[0],NodeA_num*2),dtype='float')
    # locat_npy[:,]=locat_list
    # data_list = np.concatenate((data_list, locat_npy), axis=1)

    outcome = np.array(data_list[0:190]).reshape(-1, 44)
    np.save('../Data_Tensor/X/4号树木190x44.npy', outcome)

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
    np.save('../Data3/Data_npy/实验室6号树木/6号树木501x28.npy', outcome)

def write_only_time_mean(osname):
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
    outcome = []
    for i in range(len(filenames)):
        data_list = readfile(filenames[i])
        outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 28)
    outcome=np.mean(outcome,axis=0)
    np.save('../Data3/Data_npy/实验室2号树木/locate1/2号树木501x28_mean.npy', outcome)


def writejd_only_time(osname):
    """
    编写江大数据
    """

    filenames = [name for name in os.listdir(osname)
                 if os.path.isfile(os.path.join(osname, name)) and
                 (name.endswith('.txt') and
                  (name.startswith('10')))]
    for i in range(len(filenames)):
        filenames[i] = osname + filenames[i]
    outcome=[]
    for i in range(len(filenames)):
        data_list =np.loadtxt(filenames[i],dtype='str')
        data_temp=[]
        for n in range(data_list.shape[0]):
            for m in range(n,data_list.shape[1]):
                data_temp.append("".join(data_list[n][m].split('碌s')))
    outcome = np.array(data_temp,dtype='float')
    outcome=outcome/1000
    np.save('../Data3/Data_npy/江大雪松/1x45_10sensor.npy', outcome)

class median_Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.last = None

# 链表：节点管理类
class Link:
    def __init__(self):
        # 链表的表头
        self.head = median_Node(-1)

    def length(self):
        count = -1
        temp = self.head
        while temp:
            count += 1
            temp = temp.next
        return count

    def show(self):
        temp = self.head.next
        while temp:
            print(temp.data)
            temp = temp.next

    def median(self, length):
        len_half = int(length / 2)
        node_list = []
        temp = self.head.next
        while temp:
            node_list.append(temp.data)
            temp = temp.next
        node_list.sort()
        return node_list[len_half]

    def add(self, item,data_length):
        node = median_Node(item)
        if (self.head.next == None):
            node.last = self.head
            self.head.next = node
            return
        temp = self.head
        while (temp.next != None):
            temp = temp.next
        node.last = temp
        temp.next = node
        if (self.length() > data_length):
            self.head.next = self.head.next.next
            self.head.next.next.last = self.head

    def suit_data(self, length, yuzhi):
        len_half = int(length / 2)
        node_list = []
        temp = self.head.next
        while temp:
            node_list.append(temp.data)
            temp = temp.next
        node_list.sort()
        num = len_half
        for num in range(len_half, 0, -1):
            if num == 0:
                break
            if (node_list[num] - node_list[num - 1] <= yuzhi):
                continue
            break
        return node_list[num]

    def my_mean(self,length):
        node_list = []
        temp = self.head.next
        while temp:
            node_list.append(temp.data)
            temp = temp.next
        np_node=np.array(node_list,dtype='float')
        data_mean=np.mean(np_node)
        return data_mean

def write_only_time_median(osname):
    """
        编写1-3号树木的输入x信息，秩序改写osname和np.save的文件名
        """
    data_length=10
    filenames = [name for name in os.listdir(osname)
                 if os.path.isfile(os.path.join(osname, name)) and
                 (name.endswith('树莓派.txt') or
                  name.startswith('20') or
                  name.startswith('手按'))]
    for i in range(len(filenames)):
        filenames[i] = osname + filenames[i]
    outcome = []
    for i in range(len(filenames)):
        data_list = readfile(filenames[i])
        outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 28)
    link_A = [[] for i in range(NodeA_num)]
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            link_A[i].append(Link())
    count = int(outcome.shape[0] / data_length)
    datatemp = [[] for i in range(count)]#取中值后加上选取最小值
    data_test=[[] for i in range(count)]#取中值
    data_mean=[[] for i in range(count)]#取平均值
    count = 0
    for times in range(int(outcome.shape[0] / data_length)):
        for dl in range(data_length):
            for r in range(NodeA_num - 1):
                for i in range(r + 1, NodeA_num):
                    sum = 0
                    for temp in range(r - 0):
                        sum += 7 - temp
                    time_difference = outcome[times * data_length + dl][i - r + sum - 1]
                    link_A[r][i].add(time_difference, data_length)
        for i in range(NodeA_num - 1):
            for j in range(i + 1, NodeB_num):
                # datatemp[count].append(str(link_A[i][j].suit_data(data_length,yuzhi)))
                data_test[count].append(str(link_A[i][j].median(data_length)))
                # data_mean[count].append(str(link_A[i][j].my_mean(data_length)))
        for r in range(NodeA_num - 1):
            for i in range(r + 1, NodeA_num):
                link_A[r][i] = Link()
        count += 1
    # datatemp = np.array(datatemp,dtype='float')
    data_test = np.array(data_test,dtype='float')
    # data_mean = np.array(data_mean,dtype='float')
    np.save('../Data3/Data_npy/实验室2号树木/locate1/2号树木50x28_median.npy', data_test)
    # print(np.sum(np.std(data_test, axis=0)-np.std(datatemp, axis=0)))
    # print(np.sum(np.std(data_test, axis=0)-np.std(data_mean, axis=0)))
    pass

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
    NodeA_num=8
    # readnpy_writex()
    # write_only_time(osname = '../Data3/Data_file/实验室6号树木/')
    write_only_time_mean(osname = '../Data3/Data_file/实验室2号树木/locate1/')