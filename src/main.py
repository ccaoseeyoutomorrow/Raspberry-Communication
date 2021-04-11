import socket
import time,warnings
from def_repet9 import show_average as show_average9
from def_repet9 import def_show as show9
from def_repet_RESN import show_average as show_average_resn
from def_repet_RESN_mybias import show_average as show_average_composite
from def_repet9 import average_of_show as average_show9 #层析完后取平均
from def_repet_RESN import average_of_show as average_showresn
from def_repet_RESN_mybias import average_of_show as average_showcomposite

import numpy as np



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    filename='../Data3/Data_file/实验室2号树木/'
    labelname='../Data3/label/label2.txt'
    show_average_resn(filename, labelname,70.3)
    show_average_composite(filename, labelname,70.3)
    show_average9(filename, labelname,70.3)
    # show9('../Data/实验室1号树木/树莓派01061514.txt', '../Data/实验室1号树木/location.txt','../label_Data/label1_20.txt')

    # average_showresn('../Data2/实验室3号树木/locate1/', '../label_Data/label3_20.txt')
    # average_showcomposite('../Data2/实验室3号树木/locate1/', '../label_Data/label3_20.txt')
    # average_show9('../Data2/实验室3号树木/locate1/', '../label_Data/label3_20.txt')
    # ('../Data/实验室1号树木/', '../label_Data/label1_20.txt')
    # ('../Data2/实验室3号树木/locate1/', '../label_Data/label3_20.txt')