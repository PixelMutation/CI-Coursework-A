Need to tune the logic functions to give optimal efficiency for exercise one
Give truth table, network diagrams, weights, biases etc.
Make it save the results somehow?
Plot the state-space graphs
And the network structure diagram.
Can i add these as functions to the class?
repeated training testing. move back from class to logic py


okay so things to change on the ANN

firstly could as biases to the network
tuning of the random staring values, different methods

different functions other than sigmoid
try changing to hyperbolic activation function, requires different gradient descent equation (see lecture)

use batches (part of the dataset) during training. Then test the result
avoid overtraining by doing the minimum amount of training possible
see if error is actually improving with each round 

EPOCHS https://www.baeldung.com/cs/epoch-neural-networks
basically how long it takes to use the full dataset
split into batches
one epoch is one use of the entire dataset
overfitting, generalisation, how to train without preventing it from being able to learn new things
so e.g. learning how to recognise one number really well vs all numbers equally

add batch gradient descent - so train an entire batch in one call rather than a section at a time

Could change to a CNN

Signal processing options
so maybe for the fashion dataset could use traditional means to make the image clearer, improve constrast?

on numbers that fail, can be like well if a human couldn't recognise it easily then???

use a proper framework (pytorch / tensorflow) and take advantage of hardware capabilities for training
Try out different activation functions that are easier to process

doing training using the same seeds? so not starting with random values too much

different batch approaches?
how batches help generalisation and reduce the effect of single unusual training data changing the weights too much and causing local minima

edit the high / low target used during training away from 0.01 and 0.99? why not? affects generalisation?


smaller batch sizes require larger learning rates for best performance??

training specifically on areas where it struggles as an improvement.
so training more on class 6 for fashion dataset