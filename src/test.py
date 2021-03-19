import requests
import json,time
import numpy as np

def generate(n):
    n=int(n)
    nn=int(pow(n,n))
    nnstr=np.zeros(shape=(nn,n),dtype='int')
    for j in range(n):
        tempnn=int(pow(n,n-j-1))
        count=0
        while nnstr[-1,j]==0:
            for i in range(n):
                nnstr[i*tempnn+count:i*tempnn+tempnn+count,j]=i+1
            count+=n*tempnn
    print(nnstr)



temp=1

np.savetxt('yuzhi.txt', np.array([temp]), fmt='%d')