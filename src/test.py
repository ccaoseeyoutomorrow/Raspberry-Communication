from math import pi,cos
import math,time
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # 更新
# x=[]
# y1=[]
# y2=[]
# y3=[]
# H=np.random.uniform(10e-16,10e-1,size=50)
# H=np.sort(H)
# for h in H:
#     df1=(cos(1+h)-cos(1))/h
#     df2=(cos(1-h)-cos(1))/h
#     df3=(cos(1+h)-cos(1-h))/(2*h)
#     x.append(h)
#     y1.append(df1)
#     y2.append(df2)
#     y3.append(df3)
#
# ax.plot(x, y1, 'go')
# ax.plot(x, y2, 'ro')
# ax.plot(x, y3, 'yo')
# plt.show()

for i in range(28,44,2):
    print(i)