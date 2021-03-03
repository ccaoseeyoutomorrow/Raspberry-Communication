# 缺陷复现
# 不同速度偏置算法
#分段椭圆
#增加树木边缘时间偏置,假设所有偏置根据不同位置时间不同
#计算剩余时间偏置时，假设收发时间偏置不一样
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.patches import Ellipse
import time
from numpy.linalg import lstsq

Cell_Number=100
Ray_Number=100
NodeA_num = 8  # 节点数量
NodeB_num = 8
Tree_Radius = 10  # 树木半径，单位：cm
PI = 3.141592654



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
    def __init__(self,Node_list_R,Node_list_T,file_time_name):
        self.C = np.zeros(shape=(NodeA_num, NodeB_num))  # 类似离心率
        self.B = np.zeros(shape=(NodeA_num, NodeB_num))  # 短轴
        self.Time_list = np.zeros(shape=(NodeA_num, NodeB_num))
        self.Distance_list = np.zeros(shape=(NodeA_num, NodeB_num)) # 距离，也是长轴
        self.Speed_list = np.zeros(shape=(NodeA_num, NodeB_num))
        self.bias = np.zeros(shape=(NodeA_num, NodeB_num)) # 偏置：根据线段距离圆心距离百分比
        self.bias2 = np.zeros(shape=(NodeA_num, NodeB_num)) # 偏置：1-β*β
        self.timebias =  np.zeros(shape=(NodeA_num,2))

        for i in range(NodeA_num):  # 距离list赋值
            for j in range(NodeB_num):
                if i!=j:
                    temp1=2.4 #传感器之间的误差
                    temp2=distance(Node_list_R[i], Node_list_T[j]) #传感器之间的距离
                    # 给距离赋值，单位：厘米
                    self.Distance_list[i][j] = math.sqrt(temp1*temp1+temp2*temp2)
                else:
                    self.Distance_list[i][j]=0


        # 时间list赋值
        data_list=readfile(file_time_name)
        count=0
        for i in range(NodeA_num):
            for j in range(NodeB_num):
                if i!=j:
                    # 给时间赋值，单位：毫秒
                    self.Time_list[i][j] = data_list[count]
                    count+=1
                else:
                    self.Time_list[j][i]=0


        self.Speed_list =np.divide(self.Distance_list,self.Time_list) # 速度list赋值
        for i in range(len(self.Speed_list)):
            for j in range(len(self.Speed_list)):
                temp=min(abs(i+NodeA_num-j),abs(i-j))
                biospi=(90-temp*22.5)*math.pi/180 # 90-temp*22.5°为圆周角度数，角度转化为弧度1°=π/180
                self.bias2[i][j]=1-0.2*biospi*biospi
        self.Speed_list=np.divide(self.Speed_list,self.bias2)
        # for i in range(len(self.Speed_list)):
        #     for j in range(len(self.Speed_list)):
        #         if i!=j:
        #             percentage=speed_cal(Node_list_T[i],Node_list_T[j],Node_list_T)
        #             self.bias[i][j]=percentage
        #         else:
        #             self.bias[i][j]=1

        # 计算时间补偿
        count_array=np.array([[0,1,1,0],[1,0,2,1],[1,2,2,1],[2,1,3,2],
                              [2,3,3,2],[3,2,4,3],[3,4,4,3],[4,3,5,4],
                              [4,5,5,4],[5,4,6,5],[5,6,6,5],[6,5,7,6],
                              [6,7,7,6],[7,6,0,7],[7,0,0,7],[0,7,1,0],],dtype='int')
        # count_array=np.array([[0,0,7],[1,0,1],[2,2,3],[3,3,4],[4,4,5],[5,5,6],[6,6,7],[7,0,2],
        #                [8,1,3],[9,2,4],[10,3,5],[11,4,6],[12,5,7],[13,0,6]],dtype='int')
        arrayl=count_array.shape[0]
        A = np.zeros(shape=(arrayl,16)) # 构造系数矩阵 A
        B = np.zeros(shape=arrayl).T   # 构造转置矩阵 b （这里必须为列向量）
        count=0
        for a in range(count_array.shape[0]):
            i=count_array[a][0]*2  # 接收传感器标号 0-15
            j=count_array[a][1]*2+1# 发送传感器标号 0-15
            n=count_array[a][2]*2 # 接收传感器标号 0-15
            m=count_array[a][3]*2+1 # 发送传感器标号 0-15
            print(i,j,n,m)
            # ti=t[de0] 要求的系数
            # tj=t[de1] 要求的系数
            dij=self.Distance_list[count_array[a][0]][count_array[a][1]]
            tij=self.Time_list[count_array[a][0]][count_array[a][1]]
            dnm=self.Distance_list[count_array[a][2]][count_array[a][3]]
            tnm=self.Time_list[count_array[a][2]][count_array[a][3]]
            # tn=t[n] 要求的系数
            # tm=t[m] 要求的系数
            #   d[n][m]*bias2[i][j]*t[i]+d[n][m]*bias2[i][j]*t[j]-
            #   d[i][j]*bias2[n][m]*t[n]-d[i][j]*bias2[n][m]*t[m]==
            #   d[n][m]*t[i][j]*bias2[i][j]-d[i][j]*bias2[n][m]*t[n][m]
            A[count][i]+=dnm
            A[count][j]+=dnm
            A[count][n]-=dij
            A[count][m]-=dij
            B[count]=dnm*tij-dij*tnm
            count+=1
        r = lstsq(A,B,rcond=None)  # 调用 solve 函数求解
        print(r[2])
        for i in range(14):
            if i==13:
                n1=7
                n2=1
            elif i==0:
                n1=0
                n2=0
            else:
                n1=int((i+1)/2)
                n2=(i+1)%2
            self.timebias[n1][n2]=r[0][i]

        # 对原始时间加上偏置，重新计算
        count=0
        for i in range(NodeA_num):
            for j in range(i+1,NodeB_num):
                if i!=j:
                    # 给时间赋值，单位：毫秒
                    self.Time_list[i][j]=data_list[count]-self.timebias[i][0]-self.timebias[j][1]
                    self.Time_list[j][i] = data_list[count]-self.timebias[i][0]-self.timebias[j][1]
                else:
                    self.Time_list[j][i]=0
                count+=1
        self.Speed_list =np.divide(self.Distance_list,self.Time_list) # 速度list赋值
        self.Speed_list=np.divide(self.Speed_list,self.bias2)

        # (Tij-Ti-Tj)/(Tin-Ti-Tn)=Dij/Din
        # Ti=[Din*(Tij-Tj)-Dij(Tin-Tn)]/(Din-Dij)
        for i in range(len(self.Speed_list)):
            for j in range(len(self.Speed_list)):
                if i==j:
                    self.Speed_list[i][j]=0


        Node_number_max=8
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
                self.B[i][j]=self.C[i][j]*self.Distance_list[i][j]



class RSEN():
    def __init__(self,Line,send,receive,line_minst,Nodelist):
        if send==receive:
            return
        self.num=math.ceil(Line.Distance_list[send][receive]/line_minst*2)#该直线划分成小椭圆的数量
        self.C = 0.7# Line.C[send][receive]  # 椭圆离心率
        self.L = Line.Distance_list[send][receive]/self.num/2  # 长轴长度
        self.B = self.C*self.L  # 短轴长度
        self.cirXY= np.zeros(shape=(self.num,2))# 椭圆圆心坐标点
        self.XYaxis = np.zeros(shape=(self.num,2,2)) # 椭圆长轴坐标点
        self.Speed_list = np.zeros(shape=(self.num))
        # 设置坐标点
        perx=(Nodelist[receive].x-Nodelist[send].x)/self.num
        pery=(Nodelist[receive].y-Nodelist[send].y)/self.num
        for i in range(self.num):
            self.cirXY[i][0]=Nodelist[send].x+perx*(i*2+1)/2
            self.cirXY[i][1]=Nodelist[send].y+pery*(i*2+1)/2
            self.XYaxis[i][0][0]=Nodelist[send].x+perx*i
            self.XYaxis[i][0][1]=Nodelist[send].y+pery*i
            self.XYaxis[i][1][0]=Nodelist[send].x+perx*(i+1)
            self.XYaxis[i][1][1]=Nodelist[send].y+pery*(i+1)
        # 更新小圆的速度
        for m in range(self.num):
            count=0
            sum=0
            for i in range(NodeA_num):
                for j in range(i+1,NodeB_num):
                    if ray_ellipse(Nodelist[i].x,Nodelist[i].y,Nodelist[j].x,Nodelist[j].y
                            ,self.cirXY[m][0],self.cirXY[m][1],self.L,self.B):
                        count+=1
                        sum+=Line.Speed_list[i][j]
            if count>0:
                self.Speed_list[m]=sum/count

# 判断过(x0,y0),(x1,y1)直线是否经过以(x2,y2)为圆心，ea为长轴,eb为短轴的椭圆
def ray_ellipse(x0,y0,x1,y1,x2,y2,ea,eb):
    # 直线m=(y1-y0)*x+(x0-x1)*y+x1*y0-x0*y1
    # 椭圆上任一点P(eaCOSθ+x2,ebCOSθ+y2)
    # m=(y1-y0)*(eaCOSθ+x2)+(x0-x1)*(ebCOSθ+y2)+x1*y0-x0*y1
    # m=sqrt((y1-y0)*(y1-y0)*ea*ea+(x0-x1)*(x0-x1)*eb*eb)*sin(θ+φ)+(y1-y0)*x2+(x0-x1)*y2+x1*y0-x0*y1
    # C-sqrt((y1-y0)*(y1-y0)*ea*ea+(x0-x1)*(x0-x1)*eb*eb)<=m<=C+sqrt((y1-y0)*(y1-y0)*ea*ea+(x0-x1)*(x0-x1)*eb*eb)
    C=(y1-y0)*x2+(x0-x1)*y2+x1*y0-x0*y1
    Delta=math.sqrt((y1-y0)*(y1-y0)*ea*ea+(x0-x1)*(x0-x1)*eb*eb)
    min=C-Delta
    max=C+Delta
    if min<=0 and max>=0:
        return 1
    else:
        return 0

# 找出nplist中比thres大的，第num个小的数值，xnum、ynum为list长宽
# 例子
# temp=find_minN(self.Speed_list,2,8,8,1)
def find_minN(nplist,num,xnum,ynum,thres):
    # 找出比thres大的下标号
    temp=np.where(nplist.reshape(xnum*ynum)>thres)[0]
    # 找出下标号
    temp=temp[nplist.reshape(xnum*ynum)[temp].argsort()[num]]
    return nplist[int(temp/xnum)][temp%xnum]

#网格类
class Cell():
    def __init__(self,Node_list_A):
        self.V=np.zeros(shape=(Cell_Number, Cell_Number))
        self.X=np.zeros(shape=(Cell_Number, Cell_Number))
        self.Y=np.zeros(shape=(Cell_Number, Cell_Number))
        self.inner=np.zeros(shape=(Cell_Number, Cell_Number))
        cell_length = 30/Cell_Number
        for i in range(Cell_Number):
            self.X[:, i] = (i - Cell_Number / 2) * cell_length
            self.Y[i, :] = -(i - Cell_Number / 2) * cell_length
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                if is_inner(Node_list_A,self.X[i][j],self.Y[i][j]):
                    self.V[i][j]=1
                    self.inner[i][j]=1

    # 对应力波射线进一步处理后，根据新的小椭圆进行划分
    def update_RV(self,Line_list):
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                V_sum=0;
                V_count=0;
                for n in range(NodeA_num):
                    for m in range(NodeB_num):
                        if n!=m:
                            for x in range(Line_list[n][m].num):
                                if(Ellipse_distance(Line_list[n][m].cirXY[x][0],
                                                    Line_list[n][m].cirXY[x][1],
                                                    self.X[i][j],self.Y[i][j],
                                                    Line_list[n][m].L,Line_list[n][m].B)):
                                    V_sum+=Line_list[n][m].Speed_list[x]
                                    V_count+=1
                #如果点受影响且在园内
                if V_count!=0 and self.inner[i][j]:
                    self.V[i][j]=V_sum/V_count

    # 根据原射线形成的椭圆对方块进行划分
    def updata_V(self,Line_list,Node_list_A,Node_list_B ):
        for i in range(Cell_Number):
            for j in range(Cell_Number):
                V_sum=0;
                V_count=0;
                for n in range(NodeA_num):
                    for m in range(n+1,NodeB_num):
                        if(Ellipse_distance((Node_list_A[n].x+Node_list_B[m].x)/2,
                                            (Node_list_A[n].y+Node_list_B[m].y)/2,
                                            self.X[i][j],self.Y[i][j],
                                            Line_list.Distance_list[n][m],Line_list.B[n][m])):
                            V_sum+=Line_list.Speed_list[n][m]
                            V_count+=1
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
def Ellipse_distance(Circle_X,Circle_Y,Cell_X,Cell_Y,a,b):
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

def show_plt(list_v,yuzhi,cell_inner):
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
            if cell_inner[i][j]:
                x.append(j)
                y.append(Cell_Number-i)
    ax.plot(x, y, 'go')
    x=[]
    y=[]
    for i in range(Cell_Number):
        for j in range(Cell_Number):
            if (list_v[i][j] <= yuzhi and list_v[i][j] > 10):
                x.append(j)
                y.append(Cell_Number-i)
    ax.plot(x, y, 'ro')

    plt.show()


def ultra_ray(Speed_list,Node_list_A,Node_list_B,yuzhi):
    fig, ax=plt.subplots()  # 更新
    plt.xlim(-15, 15)
    plt.ylim(-15,15)
    for i in range(NodeA_num):
        for j in range(NodeB_num):
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
    for i in range(NodeA_num):
        for j in range(NodeB_num):
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


#找到（x0,y0）到点（x1，y1）的直线与最外围射线的交点
# 返回最外围射线所代表的的传感器下标、交点x坐标、y坐标
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

# 找到（x0,y0）到点（x1，y1）与点(x2,y2)直线的交点
# 返回交点x坐标、y坐标
def inter_pointv2(x0,y0,x1,y1,x2,y2):
    # 如果斜率不存在
    if x1==x2:
        tempx=x1
        tempy=y0
    # 如果斜率为0
    elif y1==y2:
        tempx=x0
        tempy=y1
    #如果直线斜率存在且不为0
    else:
        A = (y1-y2)/(x1- x2)
        B = y1-A*y2
        # /// > 0 = ax +b -y;  对应垂线方程为 -x -ay + m = 0;(mm为系数)
        # /// > A = a; B = b;
        m = x0 + A*y0;
        # 求两直线交点坐标
        tempx=(m-A*B)/(A*A + 1)
        tempy=A*tempx+B
    return tempx,tempy

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
    file_time_name=r'D://02_soft_temp/py_temp/树莓派数据通信/56-0111-实验室数据/1627树莓派.txt'
    Node_location_A = [(0,-12),(7.354,-7.354),(10.2,0),(7.2832,7.2832),(0,10.2),(-7.071,7.071),(-10,0),(-7.2832,-7.2832)]#实验室树木
    Node_location_B = [(0,-12),(7.354,-7.354),(10.2,0),(7.2832,7.2832),(0,10.2),(-7.071,7.071),(-10,0),(-7.2832,-7.2832)]
    # Node_location_A = [(0,-14),(9.9899,-9.9899),(14,0),(9.9899,9.9899),(0,14),(-9.9899,9.9899),(-14,0),(-9.9899,-9.9899)]#江南大学1坐标
    # Node_location_B = [(0,-14),(9.9899,-9.9899),(14,0),(9.9899,9.9899),(0,14),(-9.9899,9.9899),(-14,0),(-9.9899,-9.9899)]
    Node_list_A = Node_update(Node_location_A)
    Node_list_B = Node_update(Node_location_B)
    Ultra_Line = Ultrasonic_Line(Node_list_A, Node_list_B,file_time_name)
    line_minst=find_minN(Ultra_Line.Distance_list,0,NodeA_num,NodeB_num,3)
    small_ellipse=[[]for i in range(NodeA_num)]
    for i in range(NodeA_num):
        for j in range(NodeB_num):
            small_ellipse[i].append(RSEN(Ultra_Line,i,j,line_minst,Node_list_A))
    # ultra_ellipse(small_ellipse[0][3].Speed_list, Node_list_A, Node_list_B, 0+.Distance_list, Ultra_Line.B,yuzhi)
    cell_100 = Cell(Node_list_A)
    cell_100.update_RV(small_ellipse)
    yuzhi=find_minN(cell_100.V,0,Cell_Number,Cell_Number,1)+50
    ultra_ray(Ultra_Line.Speed_list, Node_list_A, Node_list_B,yuzhi)
    # ultra_ellipse(Ultra_Line.Speed_list, Node_list_A, Node_list_B, Ultra_Line.Distance_list, Ultra_Line.B,yuzhi)
    show_plt(cell_100.V,yuzhi,cell_100.inner)

if __name__ == '__main__':
    def_show()
