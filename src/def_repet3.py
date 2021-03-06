# 缺陷复现
# 不同速度偏置算法
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.patches import Ellipse
import time

Cell_Number=100
Ray_Number=100
Node_Number_A = 8  # 节点数量
Node_Number_B = 8
Tree_Radius = 10  # 树木半径，单位：cm
PI = 3.141592654
file_time_name= r'/Data/Szhdata/手按测量.txt'
# file_time_name=r'D://02_soft_temp/py_temp/树莓派数据通信/树莓派.txt'
# file_time_name=r'C://Users/JOJO/Desktop/古城木建筑研究/实验/20201128/树莓派记录/7.txt'

#传感器位置类
class Node():  # 存放传感器位置
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

#创建传感器类list
def Node_update(Node_location):
    Node_list = []
    for i in Node_location:
        Node_list.append(Node(i[0],i[1]))
    return Node_list#返回存放Node位置的list

#读取txt文件数据
def readfile(filename):
    data_list = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            if lines!='\n':
                temp = round(float(lines), 4)
                data_list.append(temp)  # 添加新读取的数据
    data_list = np.array(data_list)  # 将数据从list类型转换为array类型。
    return data_list


# 超声波射线类，存超声波射线的传播时间、传播距离、速度等
class Ultrasonic_Line():
    def __init__(self,Node_list_R,Node_list_T):
        self.C = np.zeros(shape=(Node_Number_A, Node_Number_B))  # 类似离心率
        self.B = np.zeros(shape=(Node_Number_A, Node_Number_B))  # 短轴
        self.Time_list = np.zeros(shape=(Node_Number_A, Node_Number_B))
        self.Distance_list = np.zeros(shape=(Node_Number_A, Node_Number_B)) # 距离，也是长轴
        self.Speed_list = np.zeros(shape=(Node_Number_A, Node_Number_B))
        self.bias = np.zeros(shape=(Node_Number_A, Node_Number_B))
        self.bias2 = np.zeros(shape=(Node_Number_A, Node_Number_B))

        for i in range(Node_Number_A):  # 距离list赋值
            for j in range(Node_Number_B):
                # 给距离赋值，单位：厘米
                self.Distance_list[i][j] = distance(Node_list_R[i], Node_list_T[j])
        for i in range(Node_Number_A):  # 距离list赋值
            for j in range(Node_Number_B):
                if i!=j:
                    # 给距离赋值，单位：厘米
                    self.Distance_list[i][j] = math.sqrt(2.4*2.4+ self.Distance_list[i][j]* self.Distance_list[i][j])


        # 时间list赋值
        data_list=readfile(file_time_name)
        count=0
        for i in range(Node_Number_A):
            for j in range(i+1,Node_Number_B):
                # 给时间赋值，单位：毫秒
                self.Time_list[i][j]=data_list[count]
                self.Time_list[j][i] = data_list[count]
                count+=1

        self.Speed_list =np.divide(self.Distance_list,self.Time_list) # 速度list赋值
        for i in range(len(self.Speed_list)):
            for j in range(len(self.Speed_list)):
                temp=min(abs(i+Node_Number_A-j),abs(i-j))
                biospi=(4-temp)*22.5/180*math.pi
                self.bias2[i][j]=1-0.2*biospi*biospi

        for i in range(len(self.Speed_list)):
            for j in range(len(self.Speed_list)):
                # if abs(i-j)<=0:
                #     self.bias[i][j]=1.0
                # else:
                percentage=speed_cal(Node_list_T[i],Node_list_T[j],Node_list_T)
                self.bias[i][j]=percentage
        self.Speed_list=np.divide(self.Speed_list,self.bias2)

        Node_number_max=8
        for i in range(Node_Number_A):  # 离心率赋值
            for j in range(Node_Number_B):
                j_i = abs(j - i)
                if (j_i < (Node_number_max / 2)):
                    self.C[i][j] = 1 - j_i / (Node_number_max / 2)
                elif (j_i > (Node_number_max / 2)):
                    self.C[i][j] = 1 - (Node_number_max - j_i) / (Node_number_max / 2)
                elif (j_i == (Node_number_max / 2)):
                    self.C[i][j] = 0.1

        for i in range(Node_Number_A):  # 短轴赋值
            for j in range(Node_Number_B):
                self.B[i][j]=self.C[i][j]*self.Distance_list[i][j]

#网格类
class Cell():
    def __init__(self):
        self.V=np.zeros(shape=(Cell_Number, Cell_Number))
        self.X=np.zeros(shape=(Cell_Number, Cell_Number))
        self.Y=np.zeros(shape=(Cell_Number, Cell_Number))
        cell_length = 30/Cell_Number
        for i in range(Cell_Number):
            self.X[:, i] = (i - Cell_Number / 2) * cell_length
            self.Y[i, :] = -(i - Cell_Number / 2) * cell_length

    def updata_V(self,Line_list,Node_list_A,Node_list_B ):
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                V_sum=0;
                V_count=0;
                for n in range(Node_Number_A):
                    for m in range(Node_Number_B):
                        if(self.Ellipse_distance((Node_list_A[n].x+Node_list_B[m].x)/2,
                                                 (Node_list_A[n].y+Node_list_B[m].y)/2,
                                                 self.X[i][j],self.Y[i][j],
                                                 Line_list.Distance_list[n][m],Line_list.B[n][m])):
                            V_sum+=Line_list.Speed_list[n][m]
                            V_count+=1
                if i==86 and j==10:
                    print('1')
                #如果点受影响且在园内
                if(V_count!=0 and is_inner(Node_list_A,self.X[i][j],self.Y[i][j])):
                    self.V[i][j]=V_sum/V_count

    #判断点Cell是否在四边形内
    def rect_tangle_distance(self,X,Y):
        temp=False
        temp1=-(6.8/2.4)*(X-3.4)
        temp2=(6.8/0.9)*(X-8.3)
        if(Y>=temp1 and Y>=temp2 and Y >= 0 and Y <= 6.8):
            temp= True
        return temp

    #判断点Cell是否在椭圆内
    # Circle_X：椭圆圆心X坐标
    # Circle_Y：椭圆圆心Y坐标
    # Cell_X：点Cell的X坐标
    # Cell_Y：点Cell的Y坐标
    # a：椭圆长轴
    # b：椭圆短轴
    def Ellipse_distance(self,Circle_X,Circle_Y,Cell_X,Cell_Y,a,b):
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




def distance(X, Y):
    return math.sqrt((X.x - Y.x) * (X.x - Y.x) + (X.y - Y.y) * (X.y - Y.y))

def show_plt(list_v,yuzhi):
    fig, ax = plt.subplots()  # 更新
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime())+'show_plt')
    x=[]
    y=[]
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if(list_v[i][j]<=10):
                x.append(j)
                y.append(Cell_Number-i)
    ax.plot(x, y, 'wo')
    x=[]
    y=[]
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if (list_v[i][j] <= yuzhi and list_v[i][j] > 10):
                x.append(j)
                y.append(Cell_Number-i)
    ax.plot(x, y, 'ro')
    x=[]
    y=[]
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if (list_v[i][j] >yuzhi):
                x.append(j)
                y.append(Cell_Number-i)
    ax.plot(x, y, 'go')
    plt.show()


def ultra_ray(Speed_list,Node_list_A,Node_list_B,yuzhi):
    fig, ax=plt.subplots()  # 更新
    plt.xlim(-15, 15)
    plt.ylim(-15,15)
    for i in range(Node_Number_A):
        for j in range(Node_Number_B):
            if(Speed_list[i][j]<=yuzhi):
                color='r'
            else:
                color='g'
            ax.plot([Node_list_A[i].x, Node_list_B[j].x],[Node_list_A[i].y, Node_list_B[j].y], color=color)
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime())+'ultra_ray')
    plt.show()

def ultra_ellipse(Speed_list,Node_list_A,Node_list_B,长轴,短轴,yuzhi):
    fig, ax=plt.subplots()  # 更新
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime())+'ultra_ellipse')
    plt.xlim(-15, 15)
    plt.ylim(-15,15)
    for i in range(Node_Number_A):
        for j in range(Node_Number_B):
            if(Speed_list[i][j]<=yuzhi):
                color='red'
            else:
                color='green'
            if j==2:
                dx=Node_list_A[i].x-Node_list_B[j].x
                dy=Node_list_A[i].y-Node_list_B[j].y
                angle = np.arctan2(dy, dx)
                angle = np.degrees(angle)
                e = Ellipse(xy=((Node_list_A[i].x+Node_list_B[j].x)/2, (Node_list_A[i].y+Node_list_B[j].y)/2),
                            width=长轴[i][j], height=短轴[i][j], angle=angle,facecolor=color, alpha=0.2)
                ax.add_artist(e)
            plt.plot([Node_list_A[i].x, Node_list_B[j].x],[Node_list_A[i].y, Node_list_B[j].y], color=color)
    plt.show()

def with_no(numA,numB):
    return numA *numB >= 0

def _calulate_corss_lines(line0_pos0, line0_pos1, line1_pos0, line1_pos1):
    """
    求两条直线直接的交点
    :param line0_pos0: 第一条直接的第一个点的坐标
    :param line0_pos1: 第一条直接的第二个点的坐标
    :param line1_pos0: 第二条直接的第一个点的坐标
    :param line1_pos1: 第二条直接的第二个点的坐标
    """
    # x = (b0*c1 – b1*c0)/D
    # y = (a1*c0 – a0*c1)/D
    # D = a0*b1 – a1*b0， (D为0时，表示两直线重合)
    line0_a =line0_pos0[1] - line0_pos1[1]
    line0_b = line0_pos1[0] - line0_pos0[0]
    line0_c = line0_pos0[0] *line0_pos1[1] - line0_pos1[0] * line0_pos0[1]
    line1_a =line1_pos0[1] - line1_pos1[1]
    line1_b = line1_pos1[0] - line1_pos0[0]
    line1_c = line1_pos0[0] *line1_pos1[1] - line1_pos1[0] * line1_pos0[1]
    d = line0_a * line1_b - line1_a * line0_b
    if d == 0:
        # 重合的边线没有交点
        return None
    x = (line0_b * line1_c - line1_b * line0_c) * 1.0 / d
    y = (line0_c * line1_a - line1_c * line0_a) * 1.0 / d
    return x,y


def inter_point(x0,y0,x1,y1,Node_list):
    tempi,tempx,tempy=0,0,0
    #如果射线的斜率不存在(为无穷）
    if x0==x1:
        for i in range(len(Node_list)):
            temp1=Node_list[i].x-x0
            temp2=Node_list[i+1].x-x0
            if temp1*temp2<=0 :
                tempx=x0
                k=(Node_list[i+1].y-Node_list[i].y)/(Node_list[i+1].x-Node_list[i].x)
                tempy=k*(tempx-Node_list[i].x)+Node_list[i].y
                tempi=i
                if(with_no(tempy,y1)):
                    return tempi,tempx,tempy
    #射线的斜率存在
    else:
        k=(y1-y0)/(x1-x0)
        for i in range(len(Node_list)):
            temp1=Node_list[i].y-k*(Node_list[i].x-x0)+y0
            if i+1<=7:
                temp2=Node_list[i+1].y-k*(Node_list[i+1].x-x0)+y0
            else:
                temp2=Node_list[0].y-k*(Node_list[0].x-x0)+y0
            if temp1*temp2<=0:
                if i+1<=7:
                    tempx,tempy=_calulate_corss_lines([x0,y0],[x1,y1],[Node_list[i].x,
                                                                       Node_list[i].y],[Node_list[i+1].x,Node_list[i+1].y])
                else:
                    tempx,tempy=_calulate_corss_lines([x0,y0],[x1,y1],[Node_list[i].x,Node_list[i].y],
                                                      [Node_list[0].x,Node_list[0].y])
                tempi=i
                if(with_no(tempy,y1)):
                    return tempi,tempx,tempy

def speed_cal(Node0,Node1,Node_list):
    ray_dis_per=distance(Node0,Node1)/Ray_Number
    disx_per=(Node0.x-Node1.x)/Ray_Number
    disy_per=(Node0.y-Node1.y)/Ray_Number
    percentage=0
    for i in range(Ray_Number):
        x=Node1.x+i*disx_per
        y=Node1.y+i*disy_per
        tempi,tempx,tempy=inter_point(0,0,x,y,Node_list)
        percentage+=distance(Node(x,y),Node(0,0))/distance(Node(tempx,tempy),Node(0,0))
    return 0.5/(percentage/Ray_Number)


def def_show():
    # Node_location_A = [(0,-15),(10.8,-10.8),(15,0),(10.8,10.8),(0,15),(-10.8,10.8),(-15,0),(-10.8,-10.8)]#江南大学1坐标
    # Node_location_B = [(0,-15),(10.8,-10.8),(15,0),(10.8,10.8),(0,15),(-10.8,10.8),(-15,0),(-10.8,-10.8)]
    Node_location_A = [(0,-12),(7.354,-7.354),(10.2,0),(7.2832,7.2832),(0,10.2),(-7.071,7.071),(-10,0),(-7.2832,-7.2832)]#实验室树木
    Node_location_B = [(0,-12),(7.354,-7.354),(10.2,0),(7.2832,7.2832),(0,10.2),(-7.071,7.071),(-10,0),(-7.2832,-7.2832)]
    Node_list_A = Node_update(Node_location_A)
    Node_list_B = Node_update(Node_location_B)
    Ultra_Line = Ultrasonic_Line(Node_list_A, Node_list_B)
    cell_100 = Cell()
    cell_100.updata_V(Ultra_Line, Node_list_A, Node_list_B)
    yuzhi=np.max(cell_100.V)-27
    ultra_ray(Ultra_Line.Speed_list, Node_list_A, Node_list_B,yuzhi)
    # ultra_ellipse(Ultra_Line.Speed_list, Node_list_A, Node_list_B, Ultra_Line.Distance_list, Ultra_Line.B,yuzhi)
    show_plt(cell_100.V,yuzhi)

if __name__ == '__main__':
    def_show()
