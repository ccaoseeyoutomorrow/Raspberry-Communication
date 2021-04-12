import os,math
import numpy as np
from find_regular import show_txts_ray


locat_file = '../Data3/Data_npy/实验室5号树木/location.txt'
osname='../Data3/Data_file/实验室5号树木/210421/'
filenames = [osname + name for name in os.listdir(osname)
                     if os.path.isfile(os.path.join(osname, name)) and
                     (name.endswith('树莓派.txt') or
                      name.startswith('树莓派'))]
for name in filenames:
    max1,min1,mm1=show_txts_ray(timetxtfilename=name, locat_model=1, locat_file=locat_file, correct_model=0)
    bias1=min1/max1
    max2,min2,mm2=show_txts_ray(timetxtfilename=name, locat_model=1, locat_file=locat_file, correct_model=5)
    bias2=min2/max2
    if bias2>bias1 and bias1>0.5:
        print(mm1,mm2,name)


filename='../Data3/Data_file/实验室5号树木/210421/0422151321树莓派.txt'
print(show_txts_ray(timetxtfilename=filename, locat_model=1, locat_file=locat_file, correct_model=1))