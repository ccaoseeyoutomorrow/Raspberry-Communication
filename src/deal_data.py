import numpy as np
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

def readfile(filename):
    data_list = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            temp = round(float(lines), 4)
            data_list.append(temp)  # 添加新读取的数据
    data_list = np.array(data_list)  # 将数据从list类型转换为array类型。
    return data_list

def deal_data(data_list):
    data_length=15
    gather_number=15
    length=math.floor(data_list.shape[0]/(data_length*gather_number))
    re_list=[]
    data_temp=[]
    for i in range(length):
        data_temp=[]
        locat=i*data_length*gather_number
        for j in range(gather_number):
            temp=data_list[locat+j*data_length:locat+(j+1)*data_length]
            data_temp.append(np.median(temp))
        re_list.append(np.median(data_temp))
    re_list=np.array(re_list)
    return re_list

def deal_datatwo(data_list,dis):
    noden=28
    nodetime=10
    locat_bios=noden*nodetime*dis
    data_temp=[[]for i in range(noden)]
    for i in range(nodetime):
        for j in range(noden):
            data_temp[j].append(data_list[locat_bios+i*noden+j])
    re_list = np.median(data_temp,axis=1)
    re_list=np.array(re_list)
    print(max(re_list)-min(re_list))
    return re_list

def re_listcount(data_list):
    list_count={}
    for i in data_list:
        i = round(i, 3)
        list_count[i] = 0
    for i in data_list:
        i = round(i,3)
        list_count[i] += 1
    return list_count

def draw(list_count,name):
    fig, ax = plt.subplots()
    ax.scatter(list(list_count.keys()),list(list_count.values()))
    fig.suptitle(name)
    plt.show()

def accuracy(list_count):
    pass

if __name__ == '__main__':
    data_list={}
    # filename = 'C://Users/JOJO/Desktop/古城木建筑研究/实验/滤波算法测试/' # txt文件和当前脚本在同一目录下，所以不用写具体路径
    filename = 'record.txt'
    data_list=readfile(filename)
    for i in range(20):
        out_list=deal_datatwo(data_list,i)
        list_count=re_listcount(out_list)
        # draw(list_count,filename)
