# 创建socket对象
import socket

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
    count=501
    while True:
        if count>700:
            break
        # 接受数据
        data = conn.recv(1024)
        if not data:
            break
        filename='Data2/实验室4号树木/locate3/'+str(count)+'树莓派'+'.txt'
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