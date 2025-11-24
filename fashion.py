import pickle

from NeuralNetwork import NeuralNetwork
from mnist_functions import mnist_test, mnist_train, show_number, load_dataset

# Parameter Suggested Value
# Input Nodes 784
# Hidden Nodes 100
# Output Nodes 10
# Learning Rate 0.3

fashion=True

training_dataset=load_dataset("Datasets/fashion_mnist_train.csv")
testing_dataset=load_dataset("Datasets/fashion_mnist_test.csv")

net:NeuralNetwork
if input("Load existing network? y/n: ")=="y":
    name=input("Type Network name: ")
    with open(f"SavedNetworks/{name}.pkl", 'rb') as inp:
        net = pickle.load(inp)
else:
    net = NeuralNetwork(
        input_nodes=784, # size of the image
        hidden_nodes=int(input("How many nodes? ")),    # balance minimising nodes against maximising performance
        output_nodes=10,   # 10 digits output # could we do binary output? so 4 bits? does this help?
        learning_rate=0.3 # learning speed
    )

# implement auto save  feature perhaps? creating a folder for each test run, timestamped

print(f"Nodes: {net.hidden_nodes}, Learning Rate: {net.lr}")
for i in range(int(input("How many training epochs? "))):
    print(f"Training epoch {i}")
    mnist_train(net,training_dataset,epochs=1,batch_size=1)
    mnist_test(net,testing_dataset)

    if input("Save Network? y/n: ")=="y":
        name=input("Type Network name: ")
        with open(f"SavedNetworks/{name}.pkl", 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(net, outp, pickle.HIGHEST_PROTOCOL)
        





