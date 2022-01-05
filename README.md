# Predicting Human Randomness
This project is dedicated to predicting human randomness with machine learning.

### Introduction
This project was created to test how randomly humans can generate numbers, and how accurately AI can predict those numbers. This helps give us an understanding of how predictable human minds are to AI and machine learning. 

### Technical 
The project is primarily used to predict human number sequences in range of 1-9
#### Input Data
The input data to a trained neural network is the last three numbers of the sequence of numbers the human generated in that session.
#####
* Human generated number sequence: [1, 4, 7, 9, 8, 5] --> [9, 8, 5] NN input.
The neural network will try to predict the next number in the sequence.
#### Libraries
This project uses:
* Tensorflow
* Keras
* Numpy
### Performance
This project can predict the next number I am going to think of with 27% accuracy. Which is a 16% over guessing randomly. It doesn't use too much computing power to train the neural network either.
### Goal
The goal of this project is too explore the unexplored topic of human randomness and discover if there are any things that stand out or make an impact on how random every human is.
I would like to discover
* if intelligence is correlated to randomness
* is sleep correlated to randomness
* can machine learning learn to generate human-like number sequences
* what other demographics are correlated to randomness.

### How you can help
This project has pushed the limits of the programming and problem solving ability, if you would like to help me with this project you can help by adding your own random number sequences to the repository in /data/<your-name>/inputTrain(Test).txt. Or you can help me do experiments and try to get the accuracy of the neural network as high as possible along with many other things.
 
- Pierre
