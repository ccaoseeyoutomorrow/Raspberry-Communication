import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils import np_utils
import numpy as np


def loadmnist():
    # 训练集文件
    filepath1 = '../Data_Tensor/X/1号树木501x44.npy'
    filepath2 = '../Data_Tensor/X/2号树木501x44.npy'
    filepath3 = '../Data_Tensor/X/3号树木501x44.npy'
    filepath4 = '../Data_Tensor/X/4号树木190x44.npy'

    labelpath1='../Data_Tensor/Y/label1_20.txt'
    labelpath2='../Data_Tensor/Y/label2_20.txt'
    labelpath3='../Data_Tensor/Y/label3_20.txt'
    labelpath4='../Data_Tensor/Y/label4_20.txt'

    data1=np.array(np.load(filepath1),dtype='float')
    data2=np.array(np.load(filepath2),dtype='float')
    data3=np.array(np.load(filepath3),dtype='float')
    data4=np.array(np.load(filepath4),dtype='float')

    label1=np.zeros(shape=(data1.shape[0],20,20))
    label1[:,]=np.loadtxt(labelpath1,dtype='float')
    label2=np.zeros(shape=(data2.shape[0],20,20))
    label2[:,]=np.loadtxt(labelpath2,dtype='float')
    label3=np.zeros(shape=(data3.shape[0],20,20))
    label3[:,]=np.loadtxt(labelpath3,dtype='float')
    label4=np.zeros(shape=(data4.shape[0],20,20))
    label4[:,]=np.loadtxt(labelpath4,dtype='float')

    trainx=np.concatenate((data1,data2,data3),axis=0)
    trainy=np.concatenate((label1,label2,label3),axis=0).reshape(-1,400)
    testx=data4
    testy=label4.reshape(-1,400)


    return trainx, trainy, testx, testy


def main():
    x_train, y_train, x_test, y_test = loadmnist()
    # y_train = np_utils.to_categorical(y_train, num_classes=2)
    # y_test = np_utils.to_categorical(y_test, num_classes=2)

    model = keras.Sequential()
    model.add(keras.Input(shape=(44, 1)))  # 250x250 RGB images
    # model.add(layers.Convolution2D(  # 第一层卷积(28*28)
    #     filters=32,
    #     kernel_size=5,
    #     strides=1,
    #     padding='same',
    #     activation='relu'
    # ))
    # model.add(layers.MaxPooling2D(  # 第一层池化(14*14),相当于28除以2
    #     pool_size=2,
    #     strides=2,
    #     padding='same'
    # ))
    model.add(layers.Flatten())  # 把池化层的输出扁平化为一维数据
    model.add(Dense(600, activation='relu'))  # 第一层全连接层
    model.add(Dense(400, activation='linear'))  # 第二层全连接层
    model.summary()

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, batch_size=20, epochs=100)
    result = model.evaluate(x_test, y_test)
    print('TEST ACC:', result[1])

if __name__ == '__main__':
    main()
