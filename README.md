# Arduino-embedded-ANN-for-MNIST
An implentation of a C neural network deployed on Arduino MEGA and STM32F103C (aka "Blue/Black Pill") to recognize numbers of MNIST dataset. It is a single layer network, that is what we say for an ANN without hidden layers, based on Softmax activation function.

## Getting Started

It is possible to get the full C net implementation in /c_net/neural_network.c. This file allows you to train your model and get your weights and biases saved in c_net/net_data as binary file with extension .dat. you can also change the variable "mode" to change the main function of the program:
  if you set mode = TRAIN than once compiled the program will train the network with the given hyperparameters
  if you set mode = DISPATCH then you can use the net to make a prediction on the 784 float array put inside the flatten_image array in the else branch in the main 

### Prerequisites

You must have a C compiler, the project was written following C99 standard and you need python 3.X if you want to generate a compatible network imput from an you image made with paint, GIMP or wathever other painting tool.

## Net topology 
<p align="center">
  <img width="200" height="200" src="https://www.filepicker.io/api/file/yqw897JzTdaXecwh7cj0?policy=eyJoYW5kbGUiOiJ5cXc4OTdKelRkYVhlY3doN2NqMCIsImV4cGlyeSI6MTU4OTI4MDU2OSwiY2FsbCI6WyJyZWFkIl19&signature=1c75e8bb8b2b92f80240ea692a5f7b5676f68033bc1dcb489b53b50a2545306c">
</p>

A simple feedforward fully-connected (Dense) ANN that using Softmax map the 10 outputs within a probability distribution that sums up to 1.00. The higher propability is then mapped to 1 and all the others to 0 in order to get the categorical variable corresponding to the network output (prediction).



## Test set evaluation 

At the end of the training process the test set is evaluated and the accuracy score is reported in order to evaluate if the net overfitted the training set.


## Authors

* **Edoardo Cittadini** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)


