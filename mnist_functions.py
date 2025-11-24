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
            # store the record, the correct label, and the incorrect guess
            incorrect_records.append((record, correct_label, label))
            
    scorecard_array = np.asarray(scorecard)
    performance = (scorecard_array.sum() / scorecard_array.size) * 100
    print(f"Performance = {performance:.2f}%")
    
    # return both the score and the list of incorrect records
    return performance, incorrect_records

def mnist_train(net: NeuralNetwork, dataset, epochs=1, batch_size=64):
    # loop for the specified number of epochs
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