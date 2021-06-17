import tensorflow as tf
from tensorflow import keras
import numpy as np

def getData():
    numbers=[]
    X=[]
    y=[]
    XTest=[]
    yTest=[]
    train = open("input.txt")
    test = open("inputTest.txt")

    for line in train:
        line = line[:-1]
        for number in range(len(line)-40):
            y.append(int(line[-1]))
            line = line[:-1]
            tempX = []
            for i in line[-100:]:
                tempX.append(int(i))
            X.append(tempX)

    for line in test:
        line = line[:-1]
        for number in range(len(line)-40):
            yTest.append(int(line[-1]))
            line = line[:-1]
            tempX = []
            for i in line[-100:]:
                tempX.append(int(i))
            XTest.append(tempX)



    return np.array(X),np.array(y),np.array(XTest),np.array(yTest)

x_train, y_train,x_test, y_test = getData()

print(x_train[0],x_train[0].shape)

model = keras.Sequential()

model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_dim=x_train.shape[1]))
model.add(keras.layers.Dense(9, activation=tf.nn.softmax))

opt = keras.optimizers.Adam(learning_rate=100)


model.compile(optimizer=opt, loss='CategoricalCrossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

model.evaluate(x_test,  y_test, verbose=2)
