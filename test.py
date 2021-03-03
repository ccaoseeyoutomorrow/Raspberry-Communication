import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import LogNorm,LinearSegmentedColormap
import numpy as np
import math
import time
import os
import threading

timelist=np.zeros(shape=(5))
people = 500
class MyThread(threading.Thread):
    def __init__(self, num):
        super(MyThread, self).__init__()
        self.num = num

    def run(self):
        global people
        time.sleep(1)
        people -= 50
        print("车辆编号：%d, 当前车站人数：%d " % (self.num, people))
        print("thread id = {}".format(threading.current_thread().ident))
        self.result= people

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None



start = time.time()
vehicles=[]
for num in range(5):        # 设置车辆数
    vehicle = MyThread(num)     # 新建车辆
    vehicles.append((vehicle))

for vehicle in vehicles:        # 设置车辆数
    vehicle.start()     #启动车辆

for i in range(len(vehicles)):
    vehicles[i].join()
    timelist[i]=vehicles[i].result



end = time.time()
print("Duration time: %0.3f" % (end-start))
print(timelist)