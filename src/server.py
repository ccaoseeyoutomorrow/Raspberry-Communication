# 创建socket对象
import socket
import time
from def_repet9 import def_show as show9
from def_repet_RESN import def_show as show_resn
import numpy as np

server_receive = socket.socket()
# 确定IP
ip_port = ("0.0.0.0", 22234)
# bind()绑定
server_receive.bind(ip_port)
# listen监听
server_receive.listen(10)

count=234
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
        if max>40:
            print("数据错误")
            break
        nowtime=time.strftime("%m%d%H%M%S", time.localtime())
        filename='../Data2/实验室3号树木/locate8/'+nowtime+'树莓派.txt'
        record_file = open(filename, 'w')
        temp=str(data, encoding="utf-8")
        if temp!='\n':
            record_file.write(str(data, encoding="utf-8")+'\n')
        record_file.close()
        print(str(data, encoding="utf-8")+'\n')
        print(count)
        count+=1
        # def_show()
    # 关闭连接
    conn.close()
server_receive.close()
print("over")


