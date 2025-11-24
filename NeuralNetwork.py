# Import scipy.special for the sigmoid function expit()
import scipy.special
import numpy as np

# Neural network class definition
# This ANN does NOT use biases! Which may limit what it is capable of.
floattype=np.float64


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        self.reset()  # Initialize weights
    # activation function separated to make the class pickleable 
    def activation_function(self,x):
        return scipy.special.expit(x)

    def reset(self):
        # Initialize weights with new random values
        # may need to change method of choosing random numbers to improve chance of successful training
        # reduce chance of getting stuck in local minima
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    # Internal function to run the ANN, including intermediate outputs
    def _forward(self,inputs_array):
        # Calculate signals into hidden layer by finding dot product
        hidden_inputs = np.dot(self.wih, inputs_array)
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        return hidden_outputs,final_outputs
    
    # Train one epoch of the ANN
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = np.array(inputs_list, ndmin=2,dtype=floattype).T
        targets_array = np.array(targets_list, ndmin=2,dtype=floattype).T
        # Calculate outputs
        hidden_outputs,final_outputs=self._forward(inputs_array)
        # Calculate error
        output_errors = targets_array - final_outputs
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        np.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        np.transpose(inputs_array))
        return output_errors
    
    def train_batch(self, inputs_list, targets_list):
        inputs_array = np.array(inputs_list, ndmin=2,dtype=floattype).T
        targets_array = np.array(targets_list, ndmin=2,dtype=floattype).T
        
        hidden_outputs, final_outputs = self._forward(inputs_array)
        
        # Calculate error
        output_errors = targets_array - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # Update weights, averaged across the targets
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                    np.transpose(hidden_outputs)) / inputs_array.shape[1]
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                    np.transpose(inputs_array)) / inputs_array.shape[1]
        
        return output_errors
    
    # Query the network
    def query(self, inputs_list):
        inputs_array = np.array(inputs_list, ndmin=2,dtype=floattype).T
        _,final_outputs=self._forward(inputs_array)
        return final_outputs
    
    # Train using same targets until error becomes acceptable 
    def train_repeat(self,inputs_list,targets_list,max_error=0.1,max_epochs=1000, max_attempts=15):
        attempts=0
        while True:
            attempts+=1
            epochs=0
            while True:
                epochs+=1
                err=self.train(inputs_list,targets_list)
                if np.all(np.abs(err)<max_error):
                    print(f"Attempt {attempts}: Successfully minimised after {epochs} epochs, error {err}")
                    return
                elif epochs>max_epochs:
                    print(f"Attempt {attempts}: Failed to minimise after {epochs} epochs, error {err}")
                    break
            self.reset()
            if attempts>max_attempts:
                print("Reached 15 attempts without success")
                break





