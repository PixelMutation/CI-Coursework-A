import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork, floattype

fashion_map=[
    "Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Boot"]

# ADD FUNCTIONS TO SPLIT DATASET UP!
# maybe if it struggles on certain numbers we could train those more?

# why are these targets set like this? is it a good idea to change them?
# apparently hurts generalisation to change to somethign else

low_target=0.01 # 0.01
high_target=0.99 # 0.99 default


def load_dataset(path):
    dataset_file = open(path, 'r')
    dataset = dataset_file.readlines()
    dataset_file.close()
    return dataset

def show_number(record):
    all_values = record.split(',')
    # Take the long list of pixels (but not the label), and reshape them to a 2D array of pixels
    image_array = np.asarray(all_values[1:], dtype=floattype).reshape((28, 28))
    # Plot this 2D array as an image, use the grey colour map and don't interpolate
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

def mnist_test(net:NeuralNetwork, dataset):
    scorecard = []
    incorrect_records = []
    for record in dataset:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=floattype) / 255.0 * 0.99) + 0.01
        outputs = net.query(inputs)
        label = np.argmax(outputs)
        
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            # Store the record, the correct label, and the incorrect guess
            incorrect_records.append((record, correct_label, label))
            
    scorecard_array = np.asarray(scorecard)
    performance = (scorecard_array.sum() / scorecard_array.size) * 100
    print(f"Performance = {performance:.2f}%")
    
    # Return both the score and the list of incorrect records
    return performance, incorrect_records
# Test the network against a dataset
# def mnist_test(net:NeuralNetwork,dataset,fashion=False):
#     # Scorecard list for how well the network performs, initially empty
#     scorecard = []
#     incorrect_records = []
#     # Loop through all of the records in the test data set
#     for record in dataset:
#         # Split the record by the commas
#         all_values = record.split(',')
#         # The correct label is the first value
#         correct_label = int(all_values[0])
#         # Scale and shift the inputs
#         inputs = (np.asarray(all_values[1:], dtype=floattype) / 255.0 * 0.99) + 0.01
#         # Query the network
#         outputs = net.query(inputs)
#         # The index of the highest value output corresponds to the label
#         label = np.argmax(outputs)
#         # Append either a 1 or a 0 to the scorecard list
#         label = np.argmax(outputs)
#         # Append either a 1 or a 0 to the scorecard list
#         if (label == correct_label):
#             scorecard.append(1)
#         else:
#             scorecard.append(0)
#             incorrect_records.append((record, correct_label, label))
#     # Calculate the performance score, the fraction of correct answers
#     scorecard_array = np.asarray(scorecard)
#     performance = (scorecard_array.sum() / scorecard_array.size) * 100
#     print(f"Performance = {performance:.2f}%")

#     for i in range(10):
#         count=0
#         for _,label,_ in incorrect_records:
#             if label == i:
#                 count+=1
#         if fashion:
#             print(f"{fashion_map[i]} ({i}): {count}",end=", ")
#         else:
#             print(f"{i}: {count}",end=", ")
#     print()
#     return performance

# Train the network against a dataset
# def mnist_train(net:NeuralNetwork,dataset):
#     # Train the neural network on each training sample
#     for idx, record in enumerate(dataset):
#         print(f"Training #{idx}/{len(dataset)}   ",end='\r')
#         # Split the record by the commas
#         all_values = record.split(',')
#         # Scale and shift the inputs from 0..255 to 0.01..1
#         inputs = (np.asarray(all_values[1:], dtype=floattype) / 255.0 * 0.99) + 0.01
#         # Create the target output values (all 0.01, except the desired label which is 0.99)
#         targets = np.zeros(net.output_nodes) + low_target
#         # All_values[0] is the target label for this record
#         targets[int(all_values[0])] = high_target
#         # Train the network
#         net.train(inputs, targets)
#     print("\nTraining complete")

def mnist_train(net: NeuralNetwork, dataset, epochs=1, batch_size=64):
    # Loop for the specified number of epochs
    for epoch in range(epochs):
        np.random.shuffle(dataset)
        
        for i in range(0, len(dataset), batch_size):
            batch_list = dataset[i:i+batch_size]
            
            inputs_batch = []
            targets_batch = []
            
            for record in batch_list:
                all_values = record.split(',')
                inputs = (np.asarray(all_values[1:], dtype=floattype) / 255.0 * 0.99) + 0.01
                targets = np.zeros(net.output_nodes) + low_target
                targets[int(all_values[0])] = high_target
                
                inputs_batch.append(inputs)
                targets_batch.append(targets)

            net.train_batch(inputs_batch, targets_batch)