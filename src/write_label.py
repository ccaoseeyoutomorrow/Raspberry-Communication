# 缺陷复现
# 不同速度偏置算法
# 增加树木边缘时间偏置,假设所有偏置根据不同位置时间不同
# 计算剩余时间偏置时，假设收发时间偏置不一样
# 将射线的速度进行归一化操作
# 加入热力图
import math
import numpy as np
from def_repet9 import read_show
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import time
from numpy.linalg import lstsq
import os

Cell_Number=20





#传感器位置类
class Node():  # 存放传感器位置
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

#网格类
class Area():
    def __init__(self,radiusA,radiusB):
        self.val=np.zeros(shape=(Cell_Number, Cell_Number),dtype='int')
        self.X=np.zeros(shape=(Cell_Number, Cell_Number))
        self.Y=np.zeros(shape=(Cell_Number, Cell_Number))
        cell_length = (max(radiusA,radiusB)+1)*2/Cell_Number
        for i in range(Cell_Number):
            self.X[:, i] = (i - Cell_Number / 2) * cell_length
            self.Y[i, :] = -(i - Cell_Number / 2) * cell_length

        for i in range(Cell_Number):
            for j in range(Cell_Number):
                # 判断点是否在椭圆内
                if Ellipse_distance(0,0,self.X[i][j],self.Y[i][j],radiusA,radiusB):
                    self.val[i][j]=1

    def update_circle(self,radX1,radY1,radX2,radY2):
        """
        根据圆形缺陷，实现label编写
        :param radX1: 圆形直径的X1集合
        :param radY1: 圆形直径的Y1集合
        :param radX2: 圆形直径的X2集合
        :param radY2: 圆形直径的Y2集合
        """
        for n in range(len(radX1)):
            XC=(radX1[n]+radX2[n])/2 # 圆心X
            YC=(radY1[n]+radY2[n])/2 # 圆心Y
            x1_x2=radX1[n]-radX2[n]
            y1_y2=radY1[n]-radY2[n]
            radius=math.sqrt(x1_x2*x1_x2+y1_y2*y1_y2)/2 # 圆形半径
            for i in range(Cell_Number):
                for j in range(Cell_Number):
                    if  Ellipse_distance(XC,YC,self.X[i][j],self.Y[i][j],radius,radius): #如果在缺陷内
                        self.val[i][j]=2

    def update_polygon(self,radxys):
        """
        根据不规则缺陷，实现label编写
        :param radxys: 多边形缺陷集合
        """
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                for n in range(len(radxys)):
                    if  is_inner(radxys[n],self.X[i][j],self.Y[i][j]): #如果在缺陷内
                        self.val[i][j]=2



def Ellipse_distance(Circle_X,Circle_Y,Cell_X,Cell_Y,a,b):
    """
    判断点Cell是否在椭圆内
    :param Circle_X：椭圆圆心X坐标
    :param Circle_Y：椭圆圆心Y坐标
    :param Cell_X：点Cell的X坐标
    :param Cell_Y：点Cell的Y坐标
    :param a：椭圆长轴
    :param b：椭圆短轴
    """
    if a==0 or b==0:
        return 0
    dis=(Circle_X-Cell_X)*(Circle_X-Cell_X)/(a*a)+(Circle_Y-Cell_Y)*(Circle_Y-Cell_Y)/(b*b)
    if(dis<=1):
        return 1
    else:
        return 0

# 判断点是否在多边形内
# vertxy：多边形xy坐标点数组
# testy：点的y坐标
# testx：点的x坐标
def is_inner(vertxy,testx,testy):
    j= len(vertxy)-1
    flag=False
    for i in range(len(vertxy)):
        # 如果点在多边形两点y轴之间，且点在该两点直线的左（右）边
        temp1= (vertxy[i].y>testy)
        temp2=(vertxy[j].y>testy)
        if ( ((vertxy[i].y>testy) != (vertxy[j].y>testy)) and
                (testx < (vertxy[j].x-vertxy[i].x) * (testy-vertxy[i].y) / (vertxy[j].y-vertxy[i].y) + vertxy[i].x)):
            flag = bool(1-flag)
        j=i
    return flag;


def write_defect():
    radiusA=10 #检测树木传感器的位置长轴
    radiusB=10.2 #检测树木传感器的位置短轴
    myarea = Area(radiusA,radiusB)
    radX1=[2]
    radY1=[-7.5]
    radX2=[0]
    radY2=[-0.5]
    myarea.update_circle(radX1,radY1,radX2,radY2)
    txtfilename="../label_Data/label1_20.txt"
    np.savetxt(txtfilename, myarea.val,fmt='%d',delimiter=' ')
    read_show(txtfilename)  # 显示刚刚保存的图像
    print(myarea.val)

if __name__ == '__main__':
    write_defect()
