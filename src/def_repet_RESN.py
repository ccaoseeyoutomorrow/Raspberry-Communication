# 缺陷复现
# 设置时间段为判断依据
import math,os,warnings
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import time
from numpy.linalg import lstsq

Cell_Number = 100
Ray_Number = 100
NodeA_num = 8  # 节点数量
NodeB_num = 8
Tree_Radius = 25  # 树木最长直径 单位：cm
PI = 3.141592654


# 传感器位置类
class Node():  # 存放传感器位置
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


# 创建传感器类list
def Node_update(Node_location):
    Node_list = []
    for i in Node_location:
        Node_list.append(Node(i[0], i[1]))
    return Node_list  # 返回存放Node位置的list


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
    data_list = np.array(data_list)  # 将数据从list类型转换为array类型。
    return data_list


# 超声波射线类，存超声波射线的传播时间、传播距离、速度等
class Ultrasonic_Line():
    def __init__(self, Node_list_R, Node_list_T, data_mean):
        self.C = np.zeros(shape=(NodeA_num, NodeB_num))  # 类似离心率
        self.B = np.zeros(shape=(NodeA_num, NodeB_num))  # 短轴
        self.Time_list = np.zeros(shape=(NodeA_num, NodeB_num))
        self.Distance_list = np.zeros(shape=(NodeA_num, NodeB_num))  # 距离，也是长轴
        self.Speed_list = np.zeros(shape=(NodeA_num, NodeB_num))
        self.bias = np.zeros(shape=(NodeA_num, NodeB_num))  # 偏置：1-β*β
        self.timebias = np.zeros(shape=(NodeA_num, 2))

        for i in range(NodeA_num):  # 距离list赋值
            for j in range(NodeB_num):
                if i != j:
                    temp1 = 2.4  # 传感器之间的误差
                    temp2 = distance(Node_list_R[i], Node_list_T[j])  # 传感器之间的距离
                    # 给距离赋值，单位：厘米
                    self.Distance_list[i][j] = math.sqrt(temp1 * temp1 + temp2 * temp2)
                else:
                    self.Distance_list[i][j] = 0

        # 时间list赋值
        data_list = data_mean
        count = 0
        for i in range(NodeA_num):
            for j in range(i + 1, NodeB_num):
                if i != j:
                    # 给时间赋值，单位：毫秒
                    self.Time_list[i][j] = data_list[count]
                    self.Time_list[j][i] = data_list[count]
                else:
                    self.Time_list[j][i] = 0
                count += 1

        self.Speed_list = np.divide(self.Distance_list, self.Time_list)  # 速度list赋值
        for i in range(len(self.Speed_list)):
            for j in range(len(self.Speed_list)):
                temp = min(abs(i + NodeA_num - j), abs(i - j))
                biospi = (90 - temp * 22.5) * math.pi / 180  # 90-temp*22.5°为圆周角度数，角度转化为弧度1°=π/180
                self.bias[i][j] = 1 - 0.2 * biospi * biospi
        self.Speed_list = np.divide(self.Speed_list, self.bias)

        # 速度归一化操作
        # 找出比thres大的下标号
        temp = np.where(self.Speed_list.reshape(NodeA_num * NodeB_num) < 1000)[0]
        # 找出最大/小速度的下标号
        maxlabel = temp[self.Speed_list.reshape(NodeA_num * NodeB_num)[temp].argsort()[-1]]
        mixlabel = temp[self.Speed_list.reshape(NodeA_num * NodeB_num)[temp].argsort()[0]]
        maxspeed = self.Speed_list.reshape(NodeA_num * NodeB_num)[maxlabel]
        minspeed = self.Speed_list.reshape(NodeA_num * NodeB_num)[mixlabel]
        mm = maxspeed - minspeed
        self.yuzhi=(maxspeed*0.8-minspeed)/mm
        self.mm=mm
        self.minspeed=minspeed
        self.maxspeed=maxspeed
        count = 0
        for i in range(NodeA_num):
            for j in range(i + 1, NodeB_num):
                if i != j:
                    # 给时间赋值，单位：毫秒
                    self.Speed_list[i][j] = (self.Speed_list[i][j] - minspeed) / mm
                    self.Speed_list[j][i] = (self.Speed_list[j][i] - minspeed) / mm
                else:
                    self.Time_list[j][i] = 0
                count += 1
        Node_number_max = 8
        for i in range(NodeA_num):  # 离心率赋值
            for j in range(NodeB_num):
                j_i = abs(j - i)
                if (j_i < (Node_number_max / 2)):
                    self.C[i][j] = 1 - j_i / (Node_number_max / 2)
                elif (j_i > (Node_number_max / 2)):
                    self.C[i][j] = 1 - (Node_number_max - j_i) / (Node_number_max / 2)
                elif (j_i == (Node_number_max / 2)):
                    self.C[i][j] = 0.1

        for i in range(NodeA_num):  # 短轴赋值
            for j in range(NodeB_num):
                self.B[i][j] = self.C[i][j] * self.Distance_list[i][j]

    def Shen_updateV(self, file_time_name):
        # 时间list赋值
        data_list = readfile(file_time_name)
        # 计算时间补偿
        speed_sort = np.zeros(shape=(28), dtype='int')
        count = 0
        for i in range(8):
            for j in range(i + 1, 8):
                speed_sort[count] = i * 8 + j
                count += 1
        sorted_speed = speed_sort[self.Speed_list.reshape(8 * 8)[speed_sort].argsort()]
        count_num = 13
        count_array = np.zeros(shape=(count_num, 3), dtype='int')
        count_array1 = [[0, 0, 7], [1, 0, 1], [2, 2, 3], [3, 3, 4], [4, 4, 5], [5, 5, 6], [6, 6, 7]]
        for i in range(count_num):
            count_array[i] = [i, int(sorted_speed[-i - 2] / 8), sorted_speed[-i - 2] % 8]
        # count_array=np.array([[0,0,7],[1,0,1],[2,2,3],[3,3,4],[4,4,5],[5,5,6],[6,6,7],[7,0,2],
        #                [8,1,3],[9,2,4],[10,3,5],[11,4,6],[12,5,7],[13,0,6]],dtype='int')
        count_array = np.vstack((count_array1, count_array))
        for i in range(count_array.shape[0]):
            count_array[i][0] = i
        arrayl = count_array.shape[0]
        A = np.zeros(shape=(arrayl, 14))  # 构造系数矩阵 A
        B = np.zeros(shape=arrayl).T  # 构造转置矩阵 b （这里必须为列向量）
        de0 = 1  # 默认接收偏置标号
        de1 = 4  # 默认发送偏置标号
        del0 = 1  # 默认接收传感器位置
        del1 = 2  # 默认发送传感器位置l
        dij = self.Distance_list[del0][del1]
        bij = self.bias[del0][del1]
        tij = self.Time_list[del0][del1]
        for c, a, b in count_array:
            # 接收传感器标号
            if a == 0:
                n = 0
            else:
                n = a * 2 - 1
                # 发送传感器标号
            if b == 7:
                m = 7 * 2 - 1
            else:
                m = b * 2
            # ti=t[de0] 要求的系数
            # tj=t[de1] 要求的系数
            dnm = self.Distance_list[a][b]
            bnm = self.bias[a][b]
            tnm = self.Time_list[a][b]
            # tn=t[n] 要求的系数
            # tm=t[m] 要求的系数
            #   d[n][m]*bias[i][j]*t[i]+d[n][m]*bias[i][j]*t[j]-
            #   d[i][j]*bias[n][m]*t[n]-d[i][j]*bias[n][m]*t[m]==
            #   d[n][m]*t[i][j]*bias[i][j]-d[i][j]*bias[n][m]*t[n][m]
            A[c][de0] += dnm * bij
            A[c][de1] += dnm * bij
            A[c][n] -= dij * bnm
            A[c][m] -= dij * bnm
            B[c] = dnm * tij * bij - dij * bnm * tnm
        r = lstsq(A, B, rcond=None)  # 调用 solve 函数求解
        print(r[2])
        for i in range(14):
            if i == 13:
                n1 = 7
                n2 = 1
            elif i == 0:
                n1 = 0
                n2 = 0
            else:
                n1 = int((i + 1) / 2)
                n2 = (i + 1) % 2
            self.timebias[n1][n2] = r[0][i]

        # 对原始时间加上偏置，重新计算
        count = 0
        for i in range(NodeA_num):
            for j in range(i + 1, NodeB_num):
                if i != j:
                    # 给时间赋值，单位：毫秒
                    self.Time_list[i][j] = data_list[count] - self.timebias[i][0] - self.timebias[j][1]
                    self.Time_list[j][i] = data_list[count] - self.timebias[i][0] - self.timebias[j][1]
                else:
                    self.Time_list[j][i] = 0
                count += 1
        self.Speed_list = np.divide(self.Distance_list, self.Time_list)  # 速度list赋值
        self.Speed_list = np.divide(self.Speed_list, self.bias)


class RSEN():
    def __init__(self, Line, send, receive, line_minst, Nodelist):
        """
        :param Line: 全部射线对象
        :param send: 发送端
        :param receive: 接收端
        :param line_minst: 最短射线的长度
        :param Nodelist: 传感器坐标集合
        """
        if send == receive:
            return
        self.num = math.ceil(Line.Distance_list[send][receive] / line_minst * 2)  # 该直线划分成小椭圆的数量
        self.C = 0.8  # Line.C[send][receive]  # 椭圆离心率
        self.L = Line.Distance_list[send][receive] / self.num / 2  # 长轴长度
        self.B = self.C * self.L  # 短轴长度
        self.cirXY = np.zeros(shape=(self.num, 2))  # 椭圆圆心坐标点
        self.XYaxis = np.zeros(shape=(self.num, 2, 2))  # 椭圆长轴坐标点
        self.Speed_list = np.zeros(shape=(self.num))
        # 设置坐标点
        perx = (Nodelist[receive].x - Nodelist[send].x) / self.num
        pery = (Nodelist[receive].y - Nodelist[send].y) / self.num
        for i in range(self.num):
            self.cirXY[i][0] = Nodelist[send].x + perx * (i * 2 + 1) / 2
            self.cirXY[i][1] = Nodelist[send].y + pery * (i * 2 + 1) / 2
            self.XYaxis[i][0][0] = Nodelist[send].x + perx * i
            self.XYaxis[i][0][1] = Nodelist[send].y + pery * i
            self.XYaxis[i][1][0] = Nodelist[send].x + perx * (i + 1)
            self.XYaxis[i][1][1] = Nodelist[send].y + pery * (i + 1)
        # 更新小圆的速度
        for m in range(self.num):
            count = 0
            sum = 0
            for i in range(NodeA_num):
                for j in range(i + 1, NodeB_num):
                    if ray_ellipse(Nodelist[i].x, Nodelist[i].y, Nodelist[j].x, Nodelist[j].y
                            , self.cirXY[m][0], self.cirXY[m][1], self.L, self.B):
                        count += 1
                        sum += Line.Speed_list[i][j]
            if count > 0:
                self.Speed_list[m] = sum / count


# 判断过(x0,y0),(x1,y1)直线是否经过以(x2,y2)为圆心，ea为长轴,eb为短轴的椭圆
def ray_ellipse(x0, y0, x1, y1, x2, y2, ea, eb):
    # 直线m=(y1-y0)*x+(x0-x1)*y+x1*y0-x0*y1
    # 椭圆上任一点P(eaCOSθ+x2,ebCOSθ+y2)
    # m=(y1-y0)*(eaCOSθ+x2)+(x0-x1)*(ebCOSθ+y2)+x1*y0-x0*y1
    # m=sqrt((y1-y0)*(y1-y0)*ea*ea+(x0-x1)*(x0-x1)*eb*eb)*sin(θ+φ)+(y1-y0)*x2+(x0-x1)*y2+x1*y0-x0*y1
    # C-sqrt((y1-y0)*(y1-y0)*ea*ea+(x0-x1)*(x0-x1)*eb*eb)<=m<=C+sqrt((y1-y0)*(y1-y0)*ea*ea+(x0-x1)*(x0-x1)*eb*eb)
    C = (y1 - y0) * x2 + (x0 - x1) * y2 + x1 * y0 - x0 * y1
    Delta = math.sqrt((y1 - y0) * (y1 - y0) * ea * ea + (x0 - x1) * (x0 - x1) * eb * eb)
    min = C - Delta
    max = C + Delta
    if min <= 0 and max >= 0:
        return 1
    else:
        return 0


# 找出nplist中比thres大的，第num个小的数值，xnum、ynum为list长宽
# 例子
# temp=find_minN(self.Speed_list,2,8,8,1)
def find_minN(nplist, num, xnum, ynum, thres):
    relist = nplist.reshape(xnum * ynum)
    # 找出比thres大的下标号
    temp = np.where(relist > thres)[0]
    r = relist[temp].argsort()[num]
    # 找出下标号
    temp = temp[relist[temp].argsort()[num]]
    return nplist[int(temp / xnum)][temp % xnum]


def find_yuzhi(nplist, num, xnum, ynum):
    relist = nplist.reshape(xnum * ynum)
    # 找出比thres大的下标号
    temp = np.where(relist < 10000)[0]
    r = relist[temp].argsort()[num]
    # 找出下标号
    temp = temp[relist[temp].argsort()[num]]
    return nplist[int(temp / xnum)][temp % xnum]


# 网格类
class Cell():
    def __init__(self, radiusA, radiusB):
        self.V = np.ones(shape=(Cell_Number, Cell_Number))
        self.X = np.zeros(shape=(Cell_Number, Cell_Number))
        self.Y = np.zeros(shape=(Cell_Number, Cell_Number))
        self.inner = np.zeros(shape=(Cell_Number, Cell_Number), dtype='bool')  # 是否在园内的标号
        cell_length = (max(radiusA, radiusB) + 1) * 2 / Cell_Number
        for i in range(Cell_Number):
            self.X[:, i] = (i - Cell_Number / 2) * cell_length
            self.Y[i, :] = -(i - Cell_Number / 2) * cell_length

        for i in range(Cell_Number):
            for j in range(Cell_Number):
                # 判断点是否在椭圆内
                if Ellipse_distance(0, 0, self.X[i][j], self.Y[i][j], radiusA, radiusB):
                    self.inner[i][j] = True

    # 对应力波射线进一步处理后，根据新的小椭圆进行划分
    def update_RV(self, Line_list):
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                if not self.inner[i][j]:
                    self.V[i][j]=1.1
                    continue
                V_sum = 0
                V_count = 0
                for n in range(NodeA_num):
                    for m in range(NodeB_num):
                        if n != m:
                            for x in range(Line_list[n][m].num):
                                if (Ellipse_distance(Line_list[n][m].cirXY[x][0],
                                                     Line_list[n][m].cirXY[x][1],
                                                     self.X[i][j], self.Y[i][j],
                                                     Line_list[n][m].L, Line_list[n][m].B)):
                                    V_sum += Line_list[n][m].Speed_list[x]
                                    V_count += 1
                # 如果点受影响且在园内
                if V_count != 0:
                    self.V[i][j] = V_sum / V_count

    def sperate_jianbian(self):
        """
        将cell按照0-2进行归一化操作
        :param yuzhi:
        """
        xiabiao = np.where(self.inner.reshape(Cell_Number * Cell_Number) == True)[0]
        maxspeed = max(self.V.reshape(Cell_Number * Cell_Number)[xiabiao])
        minspeed = min(self.V.reshape(Cell_Number * Cell_Number)[xiabiao])
        mm = maxspeed - minspeed
        bias = 2 / mm
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                if self.inner[i][j] == False:
                    self.V[i][j] = -0.1
                    continue
                self.V[i][j] = (self.V[i][j] - minspeed) * bias

    def sperate(self, yuzhi):
        """
        根据阈值进行颜色变换
        :param yuzhi:
        """
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                if self.inner[i][j] == False:
                    self.V[i][j] = yuzhi + 10
                    continue
                if self.V[i][j] <= yuzhi - 80:
                    self.V[i][j] = 0

    def re_label(self, yuzhi):
        """
        根据阈值生成label
        :param yuzhi:
        """
        temp = np.zeros(shape=(Cell_Number, Cell_Number))
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                if self.inner[i][j] == False:
                    continue
                if self.V[i][j] <= yuzhi:
                    temp[i][j] = 2
                else:
                    temp[i][j] = 1
        return temp

    # 根据原射线形成的椭圆对方块进行划分
    def updata_V(self, Line_list, Node_list_A, Node_list_B):
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                V_sum = 0;
                V_count = 0;
                for n in range(NodeA_num):
                    for m in range(n + 1, NodeB_num):
                        if (Ellipse_distance((Node_list_A[n].x + Node_list_B[m].x) / 2,
                                             (Node_list_A[n].y + Node_list_B[m].y) / 2,
                                             self.X[i][j], self.Y[i][j],
                                             Line_list.Distance_list[n][m], Line_list.B[n][m])):
                            V_sum += Line_list.Speed_list[n][m]
                            V_count += 1
                # 如果点受影响且在园内
                if (V_count != 0 and is_inner(Node_list_A, self.X[i][j], self.Y[i][j])):
                    self.V[i][j] = V_sum / V_count

    # 根据原射线对小格子进行速度估计
    def updata_UV(self, V_list, Node_list_A, Node_list_B, dis, sort):
        """
        :param V_list: 速度矩阵
        :param Node_list_A: A列传感器位置集合
        :param Node_list_B: B列传感器位置集合
        :param dis: cell距离线的最短距离
        :param sort: 优先选择方式，0：最快，1：最慢，2：距离最近，3：直线的长度最短，其他：平均值
        """
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                # 判断点是否在园内
                if not self.inner[i][j]:
                    continue
                CX = self.X[i][j]
                CY = self.Y[i][j]
                V_sum = 0;
                V_count = 0;
                nptemp = [[] for i in range(7)]
                for n in range(NodeA_num):
                    for m in range(n + 1, NodeB_num):
                        distemp = pl_distance(CX, CY, Node_list_A[n].x, Node_list_A[n].y, Node_list_B[m].x,
                                              Node_list_B[m].y)
                        if distemp <= dis:
                            nptemp[0].append(V_list[n][m])  # 第一行放速度
                            nptemp[1].append(distemp)  # 第二行放距离
                            nptemp[2].append(Node_list_A[n].x)  # 第三行放射线点1的x坐标
                            nptemp[3].append(Node_list_A[n].y)  # 第四行放射线点1的y坐标
                            nptemp[4].append(Node_list_B[m].x)  # 第五行放射线点2的x坐标
                            nptemp[5].append(Node_list_B[m].y)  # 第六行放射线点2的y坐标
                            nptemp[6].append(distance(Node_list_A[n], Node_list_B[m]))  # 第七行放射线的长度
                nptemp = np.array(nptemp)
                # 如果点受影响且在园内
                if nptemp[0].shape[0] > 0:
                    label = 0
                    if sort == 0:  # 选择速度最小的值
                        label = nptemp[0].argsort()[0]
                        self.V[i][j] = nptemp[0][label]
                    elif sort == 1:  # 选择速度最慢的值
                        label = nptemp[1].argsort()[-1]
                        self.V[i][j] = nptemp[0][label]
                    elif sort == 2:  # 选择距离最近的值
                        label = nptemp[1].argsort()[0]
                        self.V[i][j] = nptemp[0][label]
                    elif sort == 2:  # 选择直线长度最短的值
                        label = nptemp[6].argsort()[0]
                        self.V[i][j] = nptemp[0][label]
                    else:  # 选择平均值
                        self.V[i][j] = np.mean(nptemp[0])

    # 判断点Cell是否在四边形内
    def rect_tangle_distance(self, X, Y):
        temp = False
        temp1 = -(6.8 / 2.4) * (X - 3.4)
        temp2 = (6.8 / 0.9) * (X - 8.3)
        if (Y >= temp1 and Y >= temp2 and Y >= 0 and Y <= 6.8):
            temp = True
        return temp

    def updata_rayV(self, V_list, Node_list_A, Node_list_B, dis, sort):
        """
        :param V_list: 速度矩阵
        :param Node_list_A: A列传感器位置集合
        :param Node_list_B: B列传感器位置集合
        :param dis: cell距离线的最短距离
        :param sort: 优先选择方式，0：最快，1：最慢，2：距离最近，3：直线的长度最短，其他：平均值
        """

        maxV=find_yuzhi(V_list,-1,8,8)
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                # 判断点是否在园内
                self.V[i][j] = 2 * maxV
                if not self.inner[i][j]:
                    continue

                CX = self.X[i][j]
                CY = self.Y[i][j]
                V_sum = 0;
                V_count = 0;
                nptemp = [[] for i in range(7)]
                for n in range(NodeA_num):
                    for m in range(n + 1, NodeB_num):
                        distemp = pl_distance(CX, CY, Node_list_A[n].x, Node_list_A[n].y, Node_list_B[m].x,
                                              Node_list_B[m].y)
                        if distemp <= dis:
                            nptemp[0].append(V_list[n][m])  # 第一行放速度
                            nptemp[1].append(distemp)  # 第二行放距离
                            nptemp[2].append(Node_list_A[n].x)  # 第三行放射线点1的x坐标
                            nptemp[3].append(Node_list_A[n].y)  # 第四行放射线点1的y坐标
                            nptemp[4].append(Node_list_B[m].x)  # 第五行放射线点2的x坐标
                            nptemp[5].append(Node_list_B[m].y)  # 第六行放射线点2的y坐标
                            nptemp[6].append(distance(Node_list_A[n], Node_list_B[m]))  # 第七行放射线的长度
                nptemp = np.array(nptemp)
                # 如果点受影响且在园内
                if nptemp[0].shape[0] > 0:
                    label = 0
                    if sort == 0:  # 选择速度最小的值
                        label = nptemp[0].argsort()[0]
                        self.V[i][j] = nptemp[0][label]
                    elif sort == 1:  # 选择速度最慢的值
                        label = nptemp[1].argsort()[-1]
                        self.V[i][j] = nptemp[0][label]
                    elif sort == 2:  # 选择距离最近的值
                        label = nptemp[1].argsort()[0]
                        self.V[i][j] = nptemp[0][label]
                    elif sort == 3:  # 选择直线长度最短的值
                        label = nptemp[6].argsort()[0]
                        self.V[i][j] = nptemp[0][label]
                    else:  # 选择平均值
                        self.V[i][j] = np.mean(nptemp[0])


# 计算点P到点A，点B形成的直线的距离
def pl_distance(px, py, ax, ay, bx, by):
    # 对于两点坐标为同一点时,返回点与点的距离
    if ax == bx and ay == by:
        point_array = np.array((px, py))
        point1_array = np.array((ax, ay))
        return np.linalg.norm(point_array - point1_array)
    # 计算直线的三个参数
    A = by - ay
    B = ax - bx
    C = (ay - by) * ax + (bx - ax) * ay
    # 根据点到直线的距离公式计算距离
    distance = np.abs(A * px + B * py + C) / (np.sqrt(A ** 2 + B ** 2))
    return distance


def Ellipse_distance(Circle_X, Circle_Y, Cell_X, Cell_Y, a, b):
    """
    判断点Cell是否在椭圆内
    :param Circle_X：椭圆圆心X坐标
    :param Circle_Y：椭圆圆心Y坐标
    :param Cell_X：点Cell的X坐标
    :param Cell_Y：点Cell的Y坐标
    :param a：椭圆长轴
    :param b：椭圆短轴
    """
    if a == 0 or b == 0:
        return 0
    dis = (Circle_X - Cell_X) * (Circle_X - Cell_X) / (a * a) + (Circle_Y - Cell_Y) * (Circle_Y - Cell_Y) / (b * b)
    if (dis <= 1):
        return 1
    else:
        return 0


# 判断点是否在多边形内
# vertxy：多边形xy坐标点数组
# testy：点的y坐标
# testx：点的x坐标
def is_inner(vertxy, testx, testy):
    j = len(vertxy) - 1
    flag = False
    for i in range(len(vertxy)):
        # 如果点在多边形两点y轴之间，且点在该两点直线的左（右）边
        temp1 = (vertxy[i].y > testy)
        temp2 = (vertxy[j].y > testy)
        if (((vertxy[i].y > testy) != (vertxy[j].y > testy)) and
                (testx < (vertxy[j].x - vertxy[i].x) * (testy - vertxy[i].y) / (vertxy[j].y - vertxy[i].y) + vertxy[
                    i].x)):
            flag = bool(1 - flag)
        j = i
    return flag;


# 计算点X到点Y的距离
def distance(X, Y):
    return math.sqrt((X.x - Y.x) * (X.x - Y.x) + (X.y - Y.y) * (X.y - Y.y))


def show_plt(list_v, yuzhi, cell_inner):
    fig, ax = plt.subplots()  # 更新
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime()) + 'show_plt')
    x = []
    y = []
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if (list_v[i][j] <= 10):
                x.append(j)
                y.append(Cell_Number - i)
    ax.plot(x, y, 'wo')
    x = []
    y = []
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if cell_inner[i][j]:
                x.append(j)
                y.append(Cell_Number - i)
    ax.plot(x, y, 'go')
    x = []
    y = []
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if (list_v[i][j] <= yuzhi and list_v[i][j] > 10):
                x.append(j)
                y.append(Cell_Number - i)
    ax.plot(x, y, 'ro')
    plt.show()


def show_heatmap(list_v,red_thre=0.25,yellow_red=1.1,yellow_greed=1.1,interflag=True):
    """
    显示list_v的热力图
    :param list_v:
    :param agflag:
    """
    yellow_thre = red_thre * yellow_red
    green_thre=yellow_thre*yellow_greed
    # 红-黄-绿 带渐变
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (yellow_thre, 1.0, 1.0),
                     (green_thre, 0.0, 0.0),
                     (0.99, 0.0, 1.0),
                     (1.0, 1.0, 0.0)),

             'green': ((0.0, 1.0, 0.0),
                       (red_thre, 1.0, 1.0),
                       (0.99, 1.0, 1.0),
                       (1.0, 1.0, 0.0)),

             'blue': ((0.0, 1.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (0.99,0.0,1.0),
                      (1.0, 1.0, 0.0)),
             }
    # 红-黄-绿 无渐变
    cdict1 = {'red': ((0.0, 1.0, 1.0),
                      (0.01, 1.0, 1.0),
                      (yellow_thre, 1.0, 0.0),
                      (0.99, 0.0, 1.0),
                      (1, 1.0, 1.0)),

              'green': ((0.0, 1.0, 1.0),
                        (0.01, 1.0, 0.0),
                        (red_thre, 0.0, 1.0),
                        (1, 1.0, 1.0)),

              'blue': ((0.0, 1.0, 1.0),
                       (0.01, 1.0, 0.0),
                       (0.99, 0.0, 1.0),
                       (1, 1.0, 1.0)),
              }

    cmap_name = 'my_list'
    fig, axs = plt.subplots(figsize=(18, 18))
    fig.suptitle('heatmap', size=50)
    blue_red1 = LinearSegmentedColormap(cmap_name, cdict)
    plt.register_cmap(cmap=blue_red1)
    plt.tick_params(labelsize=50) #设置坐标轴字体大小
    if interflag:
        im1 = axs.imshow(list_v, cmap=blue_red1,interpolation='bicubic')
    else:
        im1 = axs.imshow(list_v, cmap=blue_red1)
    cb=fig.colorbar(im1, ax=axs)  # 在图旁边加上颜色bar
    cb.ax.tick_params(labelsize=50)  #设置色标刻度字体大小。
    plt.show()


def ultra_ray(Speed_list, Node_list_A, Node_list_B, yuzhi):
    fig, ax = plt.subplots()  # 更新
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            if (Speed_list[i][j] <= yuzhi):
                color = 'r'
            else:
                color = 'g'
            ax.plot([Node_list_A[i].x, Node_list_B[j].x], [Node_list_A[i].y, Node_list_B[j].y], color=color)
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime()) + 'ultra_ray')
    plt.show()


def ultra_ellipse(Node_list_A, Node_list_B, 长轴, 短轴):
    """
    绘制椭圆
    :param Speed_list:
    :param Node_list_A:
    :param Node_list_B:
    :param 长轴:
    :param 短轴:
    :param yuzhi:
    """
    fig, ax = plt.subplots()  # 更新
    fig.suptitle(time.strftime("%m%d%H%M%S", time.localtime()) + 'ultra_ellipse')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            color = 'green'
            if j == 2:
                dx = Node_list_A[i].x - Node_list_B[j].x
                dy = Node_list_A[i].y - Node_list_B[j].y
                angle = np.arctan2(dy, dx)
                angle = np.degrees(angle)
                e = Ellipse(xy=((Node_list_A[i].x + Node_list_B[j].x) / 2, (Node_list_A[i].y + Node_list_B[j].y) / 2),
                            width=长轴[i][j], height=短轴[i][j], angle=angle, facecolor=color, alpha=0.2)
                ax.add_artist(e)
                # 增加椭圆
            plt.plot([Node_list_A[i].x, Node_list_B[j].x], [Node_list_A[i].y, Node_list_B[j].y], color=color)
    plt.show()


def with_no(numA, numB):
    return numA * numB >= 0


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
    line0_a = line0_pos0[1] - line0_pos1[1]
    line0_b = line0_pos1[0] - line0_pos0[0]
    line0_c = line0_pos0[0] * line0_pos1[1] - line0_pos1[0] * line0_pos0[1]
    line1_a = line1_pos0[1] - line1_pos1[1]
    line1_b = line1_pos1[0] - line1_pos0[0]
    line1_c = line1_pos0[0] * line1_pos1[1] - line1_pos1[0] * line1_pos0[1]
    d = line0_a * line1_b - line1_a * line0_b
    if d == 0:
        # 重合的边线没有交点
        return None
    x = (line0_b * line1_c - line1_b * line0_c) * 1.0 / d
    y = (line0_c * line1_a - line1_c * line0_a) * 1.0 / d
    return x, y


# 找到（x0,y0）到点（x1，y1）的直线与最外围射线的交点
# 返回最外围射线所代表的的传感器下标、交点x坐标、y坐标
def inter_point(x0, y0, x1, y1, Node_list):
    tempi, tempx, tempy = 0, 0, 0
    # 如果射线的斜率不存在(为无穷）
    if x0 == x1:
        for i in range(len(Node_list)):
            temp1 = Node_list[i].x - x0
            temp2 = Node_list[i + 1].x - x0
            if temp1 * temp2 <= 0:
                tempx = x0
                k = (Node_list[i + 1].y - Node_list[i].y) / (Node_list[i + 1].x - Node_list[i].x)
                tempy = k * (tempx - Node_list[i].x) + Node_list[i].y
                tempi = i
                if (with_no(tempy, y1)):
                    return tempi, tempx, tempy
    # 射线的斜率存在
    else:
        k = (y1 - y0) / (x1 - x0)
        for i in range(len(Node_list)):
            temp1 = Node_list[i].y - k * (Node_list[i].x - x0) + y0
            if i + 1 <= 7:
                temp2 = Node_list[i + 1].y - k * (Node_list[i + 1].x - x0) + y0
            else:
                temp2 = Node_list[0].y - k * (Node_list[0].x - x0) + y0
            if temp1 * temp2 <= 0:
                if i + 1 <= 7:
                    tempx, tempy = _calulate_corss_lines([x0, y0], [x1, y1], [Node_list[i].x,
                                                                              Node_list[i].y],
                                                         [Node_list[i + 1].x, Node_list[i + 1].y])
                else:
                    tempx, tempy = _calulate_corss_lines([x0, y0], [x1, y1], [Node_list[i].x, Node_list[i].y],
                                                         [Node_list[0].x, Node_list[0].y])
                tempi = i
                if (with_no(tempy, y1)):
                    return tempi, tempx, tempy


# 找到（x0,y0）到点（x1，y1）与点(x2,y2)直线的交点
# 返回交点x坐标、y坐标
def inter_pointv2(x0, y0, x1, y1, x2, y2):
    # 如果斜率不存在
    if x1 == x2:
        tempx = x1
        tempy = y0
    # 如果斜率为0
    elif y1 == y2:
        tempx = x0
        tempy = y1
    # 如果直线斜率存在且不为0
    else:
        A = (y1 - y2) / (x1 - x2)
        B = y1 - A * y2
        # /// > 0 = ax +b -y;  对应垂线方程为 -x -ay + m = 0;(mm为系数)
        # /// > A = a; B = b;
        m = x0 + A * y0;
        # 求两直线交点坐标
        tempx = (m - A * B) / (A * A + 1)
        tempy = A * tempx + B
    return tempx, tempy


def speed_cal(Node0, Node1, Node_list):
    ray_dis_per = distance(Node0, Node1) / Ray_Number
    disx_per = (Node0.x - Node1.x) / Ray_Number
    disy_per = (Node0.y - Node1.y) / Ray_Number
    percentage = 0
    for i in range(Ray_Number):
        x = Node1.x + i * disx_per
        y = Node1.y + i * disy_per
        tempi, tempx, tempy = inter_point(0, 0, x, y, Node_list)
        percentage += distance(Node(x, y), Node(0, 0)) / distance(Node(tempx, tempy), Node(0, 0))
    return 0.5 / (percentage / Ray_Number)


def read_show(filename):
    a = np.loadtxt(filename, dtype=int)  # 最普通的loadtxt
    # 红-黄-绿 无渐变
    cdict = {'red': ((0.0, 0.0, 1.0),
                     (0.3, 1.0, 0.0),
                     (0.6, 0.0, 1.0),
                     (1, 1.0, 1.0)),

             'green': ((0.0, 0.0, 1.0),
                       (0.3, 1.0, 1.0),
                       (0.6, 1.0, 0.0),
                       (1, 0.0, 0.0)),

             'blue': ((0.0, 0.0, 1.0),
                      (0.3, 1.0, 0.0),
                      (0.6, 0.0, 0.0),
                      (1, 0.0, 0.0)),
             }
    cmap_name = 'defect'
    fig, axs = plt.subplots(figsize=(15, 15))
    blue_red1 = LinearSegmentedColormap(cmap_name, cdict)
    plt.register_cmap(cmap=blue_red1)
    im1 = axs.imshow(a, cmap=blue_red1)
    fig.colorbar(im1, ax=axs)  # 在图旁边加上颜色bar
    plt.show()


def compareAB(list_label, list_test):
    """
    定量算法的计算能力
    :param list_label:标签
    :param list_test: 算法生成的图像
    :return:
    """
    area_count = 0  # 横截面总数量
    defect_count = 0  # 缺陷总数量
    TP = 0  # TP-表示该区域被正确地重建为缺陷；
    FN = 0  # FN-表示该区域被错误地重建为正常木材；
    FP = 0  # FP-表示该区域被错误地重建为缺陷；
    TN = 0  # TN-则表示该区域被正确地重建为正常木材
    for i in range(list_label.shape[0]):
        for j in range(list_label.shape[1]):
            if list_label[i][j] == 2 or list_label[i][j] == 1:
                area_count += 1
            if list_label[i][j] == 2:
                defect_count += 1
                if list_test[i][j] == 2:
                    TP += 1
                else:
                    FN += 1
            elif list_test[i][j] == 2:
                FP += 1
            elif list_test[i][j] == 1:
                TN+=1
    # 准确度(Accuracy)=(TP+TN)/(TP+TN+FP+FN)
    # 精度(Precision)=TP/(TP+FP)
    # 查全率(Recall)=TP/(TP+FN)
    # Accuracy=(TP+TN)/(TP+TN+FP+FN)#正确的重建除以总面积
    # Precision=TP/(TP+FP)#正确的缺陷重建除以算法总缺陷重建
    # Recall=TP/(TP+FN)#正确的缺陷重建除以真是总缺陷面积
    # print("总共面积单元：",area_count,
    #       "\n缺陷面积单元：",defect_count,
    #       "\n计算正确的缺陷面积单元：",TP,
    #       "\n未计算的缺陷面积单元：",FN,
    #       "\n计算错误的缺陷面积单元：",FP,
    #       "\n准确度(Accuracy)：",Accuracy,
    #       "\n精度(Precision)：",Precision,
    #       "\n查全率(Recall)：",Recall)
    return area_count, defect_count, TP, FN, FP,TN



def read_txt(filename):
    a = np.loadtxt(filename, dtype=int)  # 最普通的loadtxt
    return a

def get_locationbyradius(a, b):
    """
    返回由a，b生成的list位置
    :param a: 椭圆长轴长度
    :param b: 椭圆短轴长度
    """
    locat_list = [[] for i in range(NodeA_num)]
    for i in range(NodeA_num):

        θ = i * 360 / NodeA_num - 90
        if θ == -90 or θ == 90:
            xtemp = 0
            ytemp = θ/90*b
        elif θ == 0 or θ == 180:
            xtemp = -(θ-90)/90*a
            ytemp = 0
        else:
            θ=θ/180*math.pi
            temp=math.tan(θ)
            xtemp = a*math.cos(θ)
            ytemp = b * math.sin(θ)
        locat_list[i].append(xtemp)
        locat_list[i].append(ytemp)
    locat_list = np.array(locat_list, dtype='float').reshape(8, 2)  # 将数据从list类型转换为array类型。
    return locat_list

def def_show(file_time_name,file_locat_name):
    label=read_txt('../Data/label_Data/label4_20.txt')
    locat_list = [[] for i in range(8)]
    with open(file_locat_name, 'r', encoding='utf-8') as file_to_read:
        for i in range(8):
            lines = file_to_read.readline()  # 整行读取数据
            nums = re.split(',|\n', lines)
            locat_list[i].append(nums[0])  # 添加新读取的数据
            locat_list[i].append(nums[1])  # 添加新读取的数据
    locat_list = np.array(locat_list,dtype='float').reshape(8,2)  # 将数据从list类型转换为array类型。
    # 根据位置初始化class
    Node_list_A = Node_update(locat_list)
    Node_list_B = Node_update(locat_list)
    Ultra_Line = Ultrasonic_Line(Node_list_A, Node_list_B, file_time_name)
    line_minst = find_minN(Ultra_Line.Distance_list, 0, NodeA_num, NodeB_num, 3)
    small_ellipse = [[] for i in range(NodeA_num)]
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            # pass
            small_ellipse[i].append(RSEN(Ultra_Line, i, j, line_minst, Node_list_A))
    cell_100 = Cell(10.3, 11.6)
    cell_100.update_RV(small_ellipse)  # 对应力波射线进一步处理后，根据新的小椭圆进行划分
    # 最后一个参数A
    max=0
    tempyuzhi=0
    for i in range(1,100):
        yuzhi = float(i/100)
        test = cell_100.re_label(yuzhi)  # 根据输入的阈值分成绿红两部分
        txtfilename = '../test_Data/temp'
        np.savetxt(txtfilename, test, fmt='%d', delimiter=' ')
        area_count, defect_count, TP, FN, FP,TN=compareAB(label,test)
        if (TP+FP)==0:
            continue
        Accuracy=(TP+TN)/(TP+TN+FP+FN)#正确的重建除以总面积
        Precision=TP/(TP+FP)#正确的缺陷重建除以算法总缺陷重建
        Recall=TP/(TP+FN)#正确的缺陷重建除以真是总缺陷面积
        temp=Accuracy+Precision+Recall
        if temp>max:
            max=temp
            tempyuzhi=yuzhi
    yuzhi = tempyuzhi
    print(yuzhi)
    test = cell_100.re_label(yuzhi)  # 根据输入的阈值分成绿红两部分
    txtfilename = '../test_Data/temp'
    np.savetxt(txtfilename, test, fmt='%d', delimiter=' ')
    read_show(txtfilename)  # 显示刚刚保存的图像
    area_count, defect_count, TP, FN, FP,TN=compareAB(label,test)
    Accuracy=(TP+TN)/(TP+TN+FP+FN)#正确的重建除以总面积
    Precision=TP/(TP+FP)#正确的缺陷重建除以算法总缺陷重建
    Recall=TP/(TP+FN)#正确的缺陷重建除以真是总缺陷面积
    print("总共面积单元：",area_count,
          "\n缺陷面积单元：",defect_count,
          "\n计算正确的缺陷面积单元：",TP,
          "\n未计算的缺陷面积单元：",FN,
          "\n计算错误的缺陷面积单元：",FP,
          "\n准确度(Accuracy)：",Accuracy,
          "\n精度(Precision)：",Precision,
          "\n查全率(Recall)：",Recall)

def get_XY_length(locat_list):
    list1 = [0, 4]
    list2 = [2, 6]
    list_x = locat_list[list2, :1]
    list_y = locat_list[list1, 1:2]
    list_x = np.maximum(list_x, -list_x)
    list_y = np.maximum(list_y, -list_y)
    minx = np.min(list_x)
    miny = np.min(list_y)
    return minx, miny

def defshow_heatmap(file_time_name,length):
    radius = length / 2 / math.pi  # 检测树木传感器的位置半径
    # 根据radius和Node的数量自动计算出坐标赋值
    locat_list = get_locationbyradius(radius, radius)
    # 根据位置初始化class
    Node_list_A = Node_update(locat_list)
    Node_list_B = Node_update(locat_list)
    Ultra_Line = Ultrasonic_Line(Node_list_A, Node_list_B, file_time_name)
    line_minst = find_minN(Ultra_Line.Distance_list, 0, NodeA_num, NodeB_num, 3)
    small_ellipse = [[] for i in range(NodeA_num)]
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            # pass
            small_ellipse[i].append(RSEN(Ultra_Line, i, j, line_minst, Node_list_A))
    x, y = get_XY_length(locat_list)
    cell_100 = Cell(x, y)
    cell_100.update_RV(small_ellipse)  # 对应力波射线进一步处理后，根据新的小椭圆进行划分
    show_heatmap(cell_100.V, 0)  # 显示热力图

def show_average(filelocat,label_name,length=0):
    """
    计算时间平均后的成像结果
    :param filelocat: 数据文件夹位置
    :param label_name: label文件位置
    """
    #读取时间数据
    filenames = [name for name in os.listdir(filelocat)
                 if os.path.isfile(os.path.join(filelocat, name)) and
                 (name.endswith('树莓派.txt') or
                  name.startswith('树莓派'))]
    for i in range(len(filenames)):
        filenames[i] = filelocat + filenames[i]
    outcome = []
    for i in range(len(filenames)):
        data_list = readfile(filenames[i])
        outcome.append(data_list)
    outcome = np.array(outcome).reshape(-1, 28)
    data_mean = np.mean(outcome, axis=0)

    #读取位置信息
    if length == 0:
        locat_filename = filelocat + 'location.txt'
        locat_list = [[] for i in range(8)]
        with open(locat_filename, 'r', encoding='utf-8') as file_to_read:
            for i in range(8):
                lines = file_to_read.readline()  # 整行读取数据
                nums = re.split(',|\n', lines)
                locat_list[i].append(nums[0])  # 添加新读取的数据
                locat_list[i].append(nums[1])  # 添加新读取的数据
        locat_list = np.array(locat_list, dtype='float').reshape(8, 2)  # 将数据从list类型转换为array类型。
    else:
        radius = float(length) / 2 / math.pi  # 检测树木传感器的位置半径
        # 根据radius和Node的数量自动计算出坐标赋值
        locat_list = get_locationbyradius(radius, radius)


    # 根据radius和Node的数量自动计算出坐标赋值
    # locat_list=get_locationbyradius(radiusa,radiusb)
    # 根据位置初始化class
    Node_list_A = Node_update(locat_list)
    Node_list_B = Node_update(locat_list)
    Ultra_Line = Ultrasonic_Line(Node_list_A, Node_list_B, data_mean)
    line_minst = find_minN(Ultra_Line.Distance_list, 0, NodeA_num, NodeB_num, 3)
    small_ellipse = [[] for i in range(NodeA_num)]
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            # pass
            small_ellipse[i].append(RSEN(Ultra_Line, i, j, line_minst, Node_list_A))
    x, y = get_XY_length(locat_list)
    cell_100 = Cell(x, y)
    cell_100.update_RV(small_ellipse)  # 对应力波射线进一步处理后，根据新的小椭圆进行划分

    tempi = 0
    countp=0
    sump=0
    label = read_txt(label_name)
    tempyuzhi = 0
    tempaccuracy = 0
    for i in range(1, 99):
        yuzhitemp = i / 100
        test = cell_100.re_label(yuzhitemp)  # 根据输入的阈值分成绿红两部分
        area_count, defect_count, TP, FN, FP, TN = compareAB(label, test)
        try:
            Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 正确的重建除以总面积
            Precision = TP / (TP + FP)  # 正确的缺陷重建除以算法总缺陷重建
            Recall = TP / (TP + FN)  # 正确的缺陷重建除以真是总缺陷面积
            flag = (Accuracy * 1 + Precision * 0 + Recall * 0) / (1 + 0 + 0)
            if flag > tempaccuracy:
                tempyuzhi = yuzhitemp
                tempaccuracy = flag
        except:
            pass
    show_heatmap(cell_100.V,red_thre=tempyuzhi)  # 显示热力图
    temp = (tempyuzhi * Ultra_Line.mm + Ultra_Line.minspeed) / Ultra_Line.maxspeed
    countp += 1
    sump += tempyuzhi
    print(tempyuzhi, '%.4f' % tempaccuracy,
          '%.4f' % temp)
    test = cell_100.re_label(tempyuzhi)  # 根据输入的阈值分成绿红两部分
    area_count, defect_count, TP, FN, FP, TN = compareAB(label, test)
    # read_show(test)
    try:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 正确的重建除以总面积
        Precision = TP / (TP + FP)  # 正确的缺陷重建除以算法总缺陷重建
        Recall = TP / (TP + FN)  # 正确的缺陷重建除以真是总缺陷面积
        print("总共面积单元：", area_count,
              "\n缺陷面积单元：", defect_count,
              "\n计算正确的缺陷面积单元：", TP,
              "\n未计算的缺陷面积单元：", FN,
              "\n计算错误的缺陷面积单元：", FP,
              "\n准确度(Accuracy)：", Accuracy,
              "\n精度(Precision)：", Precision,
              "\n查全率(Recall)：", Recall)
    except:
        pass

def average_of_show(filelocat,label_name,length=0):
    filenames = [name for name in os.listdir(filelocat)
                 if os.path.isfile(os.path.join(filelocat, name)) and
                 (name.endswith('树莓派.txt') or
                  name.startswith('树莓派'))]
    for i in range(len(filenames)):
        filenames[i] = filelocat + filenames[i]
    if length == 0:
        locat_filename = filelocat + 'location.txt'
        locat_list = [[] for i in range(8)]
        outcome = []
        with open(locat_filename, 'r', encoding='utf-8') as file_to_read:
            for i in range(8):
                lines = file_to_read.readline()  # 整行读取数据
                nums = re.split(',|\n', lines)
                locat_list[i].append(nums[0])  # 添加新读取的数据
                locat_list[i].append(nums[1])  # 添加新读取的数据
        locat_list = np.array(locat_list, dtype='float').reshape(8, 2)  # 将数据从list类型转换为array类型。
    else:
        radius = float(length) / 2 / math.pi  # 检测树木传感器的位置半径
        # 根据radius和Node的数量自动计算出坐标赋值
        locat_list = get_locationbyradius(radius, radius)
    Node_list_A = Node_update(locat_list)
    Node_list_B = Node_update(locat_list)
    label = read_txt(label_name)

    counta=0
    countp=0
    countr=0
    for name in filenames:
        data_list = readfile(name)
        data_time=np.array(data_list,dtype='float').reshape(28)
        Ultra_Line = Ultrasonic_Line(Node_list_A, Node_list_B, data_time)
        line_minst = find_minN(Ultra_Line.Distance_list, 0, NodeA_num, NodeB_num, 3)
        small_ellipse = [[] for i in range(NodeA_num)]
        for i in range(NodeA_num):
            for j in range(NodeB_num):
                # pass
                small_ellipse[i].append(RSEN(Ultra_Line, i, j, line_minst, Node_list_A))
        x, y = get_XY_length(locat_list)
        cell_100 = Cell(x, y)
        cell_100.update_RV(small_ellipse)  # 对应力波射线进一步处理后，根据新的小椭圆进行划分

        countp = 0
        sump = 0
        tempyuzhi = 0
        maxAccuracy = 0
        maxPrecision= 0
        maxRecall= 0
        for i in range(1, 99, 2):
            yuzhitemp = i / 100
            test = cell_100.re_label(yuzhitemp)  # 根据输入的阈值分成绿红两部分
            area_count, defect_count, TP, FN, FP, TN = compareAB(label, test)
            try:
                Accuracy = (TP + TN) / (TP + TN + FP + FN)  # 正确的重建除以总面积
                Precision = TP / (TP + FP)  # 正确的缺陷重建除以算法总缺陷重建
                Recall = TP / (TP + FN)  # 正确的缺陷重建除以真是总缺陷面积
                flag = (Accuracy * 1 + Precision * 0 + Recall * 0) / (1 + 0 + 0)
                if flag > maxAccuracy:
                    tempyuzhi = yuzhitemp
                    maxAccuracy = Accuracy
                    maxPrecision = Precision
                    maxRecall = Recall
            except:
                pass
        # temp = (tempyuzhi * Ultra_Line.mm + Ultra_Line.minspeed) / Ultra_Line.maxspeed
        counta+=maxAccuracy
        countp+=maxPrecision
        countr+=maxRecall
    Accuracymean=counta/len(filenames)
    Precisionmean=countp/len(filenames)
    Recallmean=countr/len(filenames)
    print( "\n准确度(Accuracy)：", '%.4f'%Accuracymean,
              "\n精度(Precision)：",'%.4f'%Precisionmean,
              "\n查全率(Recall)：",'%.4f'%Recallmean)

def show_ray(Speed_list,locat_list,Node_list_A):
    """
    根据坐标和数值生成一张图，利用imshow显示
    :param Speed_list:速度矩阵
    :param locat_list:矩阵
    :param Node_list_B:字典
    """
    x, y = get_XY_length(locat_list)
    cell_100 = Cell(x, y)
    cell_100.updata_rayV(Speed_list, Node_list_A, Node_list_A, 0.2, 2)  # 根据原射线对小格子进行速度估计
    show_heatmap(cell_100.V,red_thre=0.3,yellow_red=1.5,yellow_greed=1.5,interflag=False)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # file_time_name = '../Data/实验室1号树木/202012291951.txt'
    # file_locat_name='../Data/实验室1号树木/location.txt'
    # def_show(file_time_name,file_locat_name)
    # defshow_heatmap(file_time_name,68)
    show_average('../Data2/实验室3号树木/locate1/', '../label_Data/label3_20.txt')
    average_of_show('../Data2/实验室3号树木/locate1/', '../label_Data/label3_20.txt')
    # read_show('test1_szh.txt')
    # label=read_txt('label1.txt')
    # test=read_txt('test1_szh.txt')
    #
    # compareAB(label,test)
