import requests
import json,time
import numpy as np
import array
url = '175.24.147.229:2399/imsi'
data = {'imsi': '12145213', 'defect':
    '68\n0.0687\n0.1300\n0.2182\n0.2154\n0.2080\n0.1555\n0.088\n0.0618' + \
    '\n0.11\n0.1686\n0.19\n0.21\n0.1426\n0.072\n0.106\n0.156\n0.1705\n' + \
    '0.1994\n0.072\n0.1035\n0.1445\n0.1949\n0.072\n0.145\n0.169\n0.072\n' + \
    '0.1257\n0.0773\n'}
headers = {'content-type': 'application/json'}


def send(V,X,Y):
    V=np.round(V,decimals=3)
    X=np.round(X,decimals=3)
    Y=np.round(Y,decimals=3)
    dict={}
    dict['imsi']='12145213'
    dict['data']=V.tolist()
    dict['XData']=X.tolist()
    dict['YData']=Y.tolist()
    jsond=json.dumps(dict)
    while 1:
        try:
            r = requests.post(url, data=json.dumps(jsond), headers=headers)
            print(r.text)
            break
        except:
            print("发送失败")
        time.sleep(10)