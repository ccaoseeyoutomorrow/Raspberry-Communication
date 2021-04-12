# 创建socket对象
import socket
import time,sys
sys.path.append('..\\src2')
from def_repet9 import def_show as show9
from def_repet_RESN import def_show as show_resn
from src2.find_regular import show_txts_ray
from src2.defect_composite import show_npys
import numpy as np

server_receive = socket.socket()
# 确定IP
ip_port = ("0.0.0.0", 22234)
# bind()绑定
server_receive.bind(ip_port)
# listen监听
server_receive.listen(10)

locat_file = '../Data3/Data_npy/实验室3号树木/location.txt'
filename = '../Data3/Data_npy/实验室5号树木/5号树木501x28_mean.npy'
labelname = '../Data3/label/实验室3号树木/label3_irregular.txt'
count=0

while True:
    # 建立客户端链接
    # accept 接受请求链接
    conn, addr = server_receive.accept()
    while True:
        if count>500:
            exit()
        # 接受数据
        data = conn.recv(1024)
        if not data:
            break
        temp=str(data, encoding="utf-8")
        max=np.max(np.array(temp.split(),dtype='float'))
        # if max>40:
        #     print("数据错误")
        #     break
        nowtime=time.strftime("%m%d%H%M%S", time.localtime())
        # recordfilename=filename+nowtime+'树莓派.txt'
        recordfilename='../Data/temp.txt'
        record_file = open(recordfilename, 'w')
        temp=str(data, encoding="utf-8")
        if temp!='\n':
            record_file.write(str(data, encoding="utf-8")+'\n')
        record_file.close()

        show_npys(recordfilename, labelname, locat_model=1, bias_model=0, defect_model=0, locat_file=locat_file,
                  correct_model=6)
        # show9(filename,'../Data2/实验室1号树木/location.txt')
        print(str(data, encoding="utf-8")+'\n')
        print(count)
        count+=1
        # def_show()
    # 关闭连接
    conn.close()
server_receive.close()
print("over")


