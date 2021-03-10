# 创建socket对象
import socket
import time
from def_repet9 import def_show as show9
from def_repet_RESN import def_show as show_resn

server_receive = socket.socket()
# 确定IP
ip_port = ("0.0.0.0", 22234)
# bind()绑定
server_receive.bind(ip_port)
# listen监听
server_receive.listen(10)

while True:
    # 建立客户端链接
    # accept 接受请求链接
    conn, addr = server_receive.accept()
    count=0
    # location_name='../Data/实验室3号树木/location.txt'
    while True:
        if count>500:
            break
        # 接受数据
        data = conn.recv(1024)
        if not data:
            break
        nowtime=time.strftime("%m%d%H%M%S", time.localtime())
        filename='../Data2/实验室3号树木/locate2/'+nowtime+'树莓派.txt'
        # filename='../test_Data/ras_data.txt'
        record_file = open(filename, 'w')
        temp=str(data, encoding="utf-8")
        if temp!='\n':
            record_file.write(str(data, encoding="utf-8")+'\n')
        record_file.close()
        print(str(data, encoding="utf-8")+'\n')
        print(count)
        count+=1
        # show9(filename,location_name)
        # show_resn(filename,location_name)
    # 关闭连接
    conn.close()
server_receive.close()
print("over")