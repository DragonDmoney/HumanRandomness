import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
import dill
import weakref

savePath = "./data/dorian/models/"
numbersInX=3
batchSize = 71
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def getData(trainingFile,testFile, numbersInX):
    numbers=[]
    X=[]
    y=[]
    XTest=[]
    yTest=[]
    f = open(str(trainingFile))
    test=open(str(testFile))

    for line in f:
        line = line[:-1]
        for number in range(len(line)-numbersInX):
            y.append(int(line[-1]))
            line = line[:-1]
            tempX = []
            for i in line[-numbersInX:]:
                tempX.append(int(i))
            X.append(tempX)
    for line in test:
        line = line[:-1]
        for number in range(len(line)-numbersInX):
            yTest.append(int(line[-1]))
            line = line[:-1]
            tempX = []
            for i in line[-numbersInX:]:
                tempX.append(int(i))
            XTest.append(tempX)

    trainP = np.random.permutation(len(y))
    testP = np.random.permutation(len(yTest))

    return np.array(X)[trainP],np.array(y)[trainP],np.array(XTest)[testP],np.array(yTest)[testP]
    # return np.array(X),np.array(y),np.array(XTest),np.array(yTest)

x_train, y_train,x_test, y_test = getData(trainingFile="./data/dorian/inputTrain.txt", testFile="./data/dorian/inputTest.txt",numbersInX=numbersInX)

model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(numbersInX)),
  tf.keras.layers.Dense(18, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  # tf.keras.layers.Dense(numbersInX, activation=tf.nn.softmax)
  ])


opt = keras.optimizers.Adam(learning_rate=0.05)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2000)

loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)

# model.save(savePath+"/tensorFlowNN-"+str(accuracy)+"-epoch:2000")
os.mkdir(savePath+"tensorFlowNN-"+str(accuracy)+"-epochs=2000")
model.save(savePath+"tensorFlowNN-"+str(accuracy)+"-epochs=2000")
