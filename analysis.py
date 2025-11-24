import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import copy

from NeuralNetwork import NeuralNetwork
from mnist_functions import mnist_test, mnist_train, load_dataset, fashion_map

##! 1000 fashion experiments!

#  CHOOSE THE EXPERIMENT TO RUN 
# Options: "HIDDEN_NODES", "LEARNING_RATE", "EPOCHS", "DATASET_FRACTION", "BATCH_SIZE", "EPOCH_PROGRESSION"

# EXPERIMENT_TO_RUN = "HIDDEN_NODES"
# #  EXPERIMENT PARAMETERS 
# FASHION_DATASET = True
# PARTIAL_DATASET = True
# # Fixed parameters 
# FIXED_LEARNING_RATE = 0.01
# FIXED_HIDDEN_NODES = 300
# FIXED_EPOCHS = 5
# FIXED_BATCH_SIZE = 1
# # Parameters to test 
# hidden_nodes_to_test = [20, 50, 100,150,  200, 300, 400]
# learning_rates_to_test = [0.01, 0.05, 0.1, 0.2, 0.3,0.5]
# epochs_to_test = [1, 2, 3,4, 5,6,7,8,9, 10,12,15]
# dataset_fractions_to_test = [0.1, 0.25, 0.5, 0.75, 1.0]
# batch_sizes_to_test = [1, 2,4,8, 16, 32, 40,50, 64, 128]


# FIXED_DATASET_FRACTION = 1.0
##! Full fashion experiments

EXPERIMENT_TO_RUN = f"LEARNING_RATE"

#  EXPERIMENT PARAMETERS (Modify these to configure the tests) 

# General settings
FASHION_DATASET = True
PARTIAL_DATASET = False # use the entire thing now

# NEXT TEST: try minimum number of nodes to get a high score

# Fixed parameters (used as constants in experiments)
FIXED_LEARNING_RATE = 0.2 # 0.2 also good??
FIXED_HIDDEN_NODES = 400  # 150 best?? 400 #400 # 100 sweet spot of efficiency, 200 best really. arguably 80 is okay
FIXED_EPOCHS = 1
FIXED_BATCH_SIZE = 8 # 10
FIXED_DATASET_FRACTION = 1.0

# Parameters to test (the values that will be varied in each experiment)
hidden_nodes_to_test = [10,20,30,40,50,60,70,80,90, 100,150,200,250, 300,350, 400, 500,600,800]
learning_rates_to_test = [0.01, 0.02,0.03,0.04,0.05,0.06,0.08, 0.1,0.15, 0.2,0.25, 0.3,0.35,0.4,0.45,0.5,0.6]
epochs_to_test = range(1,30)
dataset_fractions_to_test = [0.1, 0.25, 0.5, 0.75, 1.0]
batch_sizes_to_test = [1,2,4,6,8,10,12,14,16,18,20,25,50,75,100,200,300,400]

#! Full mnist experiments

# EXPERIMENT_TO_RUN = f"EPOCH_PROGRESSION"

# # EXPERIMENT PARAMETERS

# # General settings
# FASHION_DATASET = False
# PARTIAL_DATASET = False # whether to use the entire thing

# # Fixed parameters 
# FIXED_LEARNING_RATE = 0.6 # 0.6
# FIXED_HIDDEN_NODES = 600 # 100 sweet spot of efficiency, 200 best really. arguably 80 is okay
# FIXED_EPOCHS = 1
# FIXED_BATCH_SIZE = 8


# # Parameters to test 
# hidden_nodes_to_test = [1,5,10,20,30,40,50,60,70,80,90, 100,200, 300, 400, 500,600,800]
# learning_rates_to_test = [0.01, 0.02,0.03,0.04,0.05,0.06,0.08, 0.1,0.15, 0.2,0.25, 0.3,0.4,0.5,0.6,0.8,0.9,1,1.5]
# epochs_to_test = range(1,16)
# dataset_fractions_to_test = [0.1, 0.25, 0.5, 0.75, 1.0]
# batch_sizes_to_test = [1,2,4,8,10,15, 25,50,100,200,400]

FIXED_DATASET_FRACTION = 1.0
def analyse_best_network(best_result, fashion_map, save_path):
    print("\n Analysing Best Performing Network ")
    net, performance, incorrect_records = best_result
    print(f"Best performance was {performance:.2f}% with {net.hidden_nodes} nodes, LR={net.lr}, epochs={net.epochs}, batch_size={net.batch_size}")

    if not incorrect_records:
        print("No incorrect records to analyse. Perfect score!")
        return

    incorrect_counts = np.zeros(10)
    for _, correct_label, _ in incorrect_records:
        incorrect_counts[correct_label] += 1

    plt.figure(figsize=(12, 6))
    if FASHION_DATASET:
        plt.bar(fashion_map, incorrect_counts)
    else:
        plt.bar(range(10), incorrect_counts)
    plt.title('Count of Incorrect Classifications per Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Errors')
    if FASHION_DATASET:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(range(10))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'incorrect_classifications_barchart.png'))
    print(f"Saved incorrect classifications bar chart to {save_path}")
    # plt.close()
    # plt.show()

    fig, axes = plt.subplots(5, 10, figsize=(15, 8))
    fig.suptitle('Top 5 Misclassified Images per Category', fontsize=16)

    for i in range(10):
        if FASHION_DATASET:
            axes[0, i].set_title(fashion_map[i], fontsize=10)
        else:
            axes[0, i].set_title(i, fontsize=10)
        category_errors = [r for r in incorrect_records if r[1] == i]
        for j in range(5):
            ax = axes[j, i]
            ax.axis('off')
            if j < len(category_errors):
                record, _, predicted_label = category_errors[j]
                all_values = record.split(',')
                image_array = np.asarray(all_values[1:], dtype=np.float64).reshape((28, 28))
                ax.imshow(image_array, cmap='Greys', interpolation='None')
                if FASHION_DATASET:
                    ax.text(13.5, 32, f"-> {fashion_map[predicted_label]}", fontsize=8, color='red', ha='center')
                else:
                    ax.text(13.5, 32, f"-> {predicted_label}", fontsize=8, color='red', ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, 'incorrect_classification_examples.png'))
    print(f"Saved grid of incorrect image examples to {save_path}")
    # plt.close()
    # plt.show()

def train_and_test_net(hidden_nodes, lr, epochs, batch_size, training_dataset, testing_dataset):
    net = NeuralNetwork(input_nodes=784, hidden_nodes=hidden_nodes, output_nodes=10, learning_rate=lr)
    # Store training params in net object for later reference
    net.epochs = epochs
    net.batch_size = batch_size
    
    mnist_train(net, training_dataset, epochs=epochs, batch_size=batch_size)
    performance, incorrect_records = mnist_test(net, testing_dataset)
    return (net, performance, incorrect_records)

def run_experiment(exp_name, worker_args, x_values, x_label, plot_title_template, save_path):
    print(f"\n{'='*20}\n Running Experiment: {exp_name} \n{'='*20}")
    
    with mp.Pool() as pool:
        results = pool.starmap(train_and_test_net, worker_args)
    
    performance_scores = [r[1] for r in results]
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, performance_scores, marker='o', linestyle='--')
    plt.title(plot_title_template)
    plt.xlabel(x_label)
    plt.ylabel('Performance (%)')
    plt.grid(True)
    plot_filename = f"performance_vs_{x_label.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_path, plot_filename))
    print(f"\nSaved overall performance plot to {save_path}")
    # plt.close()
    # plt.show()

    if results:
        best_network_result = max(results, key=lambda item: item[1])
        analyse_best_network(best_network_result, fashion_map, save_path)

    print(f"\n Saving Networks for {exp_name} ")
    if input(f"Save all trained networks from the '{exp_name}' batch? (y/n): ").lower() == 'y':
        for net, _, _ in results:
            filename = f"net_{net.hidden_nodes}h_{net.lr}lr_{net.epochs}e_{net.batch_size}b.pkl"
            with open(os.path.join(save_path, filename), 'wb') as outp:
                pickle.dump(net, outp, pickle.HIGHEST_PROTOCOL)
            print(f"Saved network to {os.path.join(save_path, filename)}")

if __name__ == "__main__":
    mp.freeze_support()

    test_name = input("Enter a name for this test run (e.g., 'tuning_epochs'): ")
    save_path = os.path.join("Tests", test_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"All results will be saved in: {save_path}")

    prefix=('fashion_' if FASHION_DATASET else '')
    suffix=(("_1000" if FASHION_DATASET else "_100") if PARTIAL_DATASET else "")
    training_dataset_path = f"Datasets/{prefix}mnist_train{suffix}.csv"
    testing_dataset_path = f"Datasets/{prefix}mnist_test.csv"
    
    full_training_dataset = load_dataset(training_dataset_path)
    testing_dataset = load_dataset(testing_dataset_path)

    #  Experiment Setup 
    
    if EXPERIMENT_TO_RUN == "HIDDEN_NODES":
        training_data = full_training_dataset[:int(len(full_training_dataset) * FIXED_DATASET_FRACTION)]
        worker_args = [(nodes, FIXED_LEARNING_RATE, FIXED_EPOCHS, FIXED_BATCH_SIZE, training_data, testing_dataset) for nodes in hidden_nodes_to_test]
        run_experiment("Performance vs. Hidden Nodes", worker_args, hidden_nodes_to_test, "Number of Hidden Nodes", f'Performance vs. Hidden Nodes (lr={FIXED_LEARNING_RATE},batch={FIXED_BATCH_SIZE},epochs={FIXED_EPOCHS})', save_path)

    elif EXPERIMENT_TO_RUN == "LEARNING_RATE":
        training_data = full_training_dataset[:int(len(full_training_dataset) * FIXED_DATASET_FRACTION)]
        worker_args = [(FIXED_HIDDEN_NODES, lr, FIXED_EPOCHS, FIXED_BATCH_SIZE, training_data, testing_dataset) for lr in learning_rates_to_test]
        run_experiment("Performance vs. Learning Rate", worker_args, learning_rates_to_test, "Learning Rate", f'Performance vs. Learning Rate (n={FIXED_HIDDEN_NODES},batch={FIXED_BATCH_SIZE},epochs={FIXED_EPOCHS})', save_path)

    elif EXPERIMENT_TO_RUN == "EPOCHS":
        training_data = full_training_dataset[:int(len(full_training_dataset) * FIXED_DATASET_FRACTION)]
        worker_args = [(FIXED_HIDDEN_NODES, FIXED_LEARNING_RATE, epochs, FIXED_BATCH_SIZE, training_data, testing_dataset) for epochs in epochs_to_test]
        run_experiment("Performance vs. Number of Epochs", worker_args, epochs_to_test, "Number of Epochs", f'Performance vs. Epochs (n={FIXED_HIDDEN_NODES}, lr={FIXED_LEARNING_RATE},batch={FIXED_BATCH_SIZE})', save_path)

    elif EXPERIMENT_TO_RUN == "DATASET_FRACTION":
        results = []
        for fraction in dataset_fractions_to_test:
            print(f"\n Running with dataset fraction: {fraction*100:.0f}% ")
            training_data = full_training_dataset[:int(len(full_training_dataset) * fraction)]
            # Note: This experiment runs sequentially as it modifies the dataset for each run
            results.append(train_and_test_net(FIXED_HIDDEN_NODES, FIXED_LEARNING_RATE, FIXED_EPOCHS, FIXED_BATCH_SIZE, training_data, testing_dataset))
        
        # Manual plotting and analysis for this sequential experiment
        performance_scores = [r[1] for r in results]
        plt.figure(figsize=(10, 6))
        plt.plot(dataset_fractions_to_test, performance_scores, marker='o', linestyle='--')
        plt.title(f'Performance vs. Dataset Fraction (Nodes={FIXED_HIDDEN_NODES}, LR={FIXED_LEARNING_RATE})')
        plt.xlabel("Dataset Fraction")
        plt.ylabel('Performance (%)')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'performance_vs_dataset_fraction.png'))
        # plt.close()
        # plt.show()

        if results:
            best_network_result = max(results, key=lambda item: item[1])
            analyse_best_network(best_network_result, fashion_map, save_path)

    elif EXPERIMENT_TO_RUN == "BATCH_SIZE":
        training_data = full_training_dataset[:int(len(full_training_dataset) * FIXED_DATASET_FRACTION)]
        worker_args = [(FIXED_HIDDEN_NODES, FIXED_LEARNING_RATE, FIXED_EPOCHS, batch_size, training_data, testing_dataset) for batch_size in batch_sizes_to_test]
        run_experiment("Performance vs. Batch Size", worker_args, batch_sizes_to_test, "Batch Size", f'Performance vs. Batch Size (n={FIXED_HIDDEN_NODES}, lr={FIXED_LEARNING_RATE},epochs={FIXED_EPOCHS})', save_path)
    elif EXPERIMENT_TO_RUN == "EPOCH_PROGRESSION":
        print(f"\n{'='*20}\n--- Running Experiment: Epoch Progression ---\n{'='*20}")
        
        net = NeuralNetwork(
            input_nodes=784,
            hidden_nodes=FIXED_HIDDEN_NODES,
            output_nodes=10,
            learning_rate=FIXED_LEARNING_RATE
        )
        
        # --- CHANGE 1: Create lists for both training and validation performance ---
        training_performance_over_time = []
        validation_performance_over_time = []
        epoch_numbers = list(range(1, max(epochs_to_test) + 1))

        best_validation_performance = -1.0
        best_net_state = None
        best_result_tuple = None
        best_epoch = 0

        for epoch in epoch_numbers:
            print(f"--- Training Epoch {epoch}/{max(epochs_to_test)} ---")
            
            mnist_train(net, full_training_dataset, epochs=1, batch_size=FIXED_BATCH_SIZE)
            net.epochs = epoch
            net.batch_size = FIXED_BATCH_SIZE
            # --- CHANGE 2: Test performance on BOTH datasets ---
            print("  Testing on validation set...")
            validation_performance, incorrect_records = mnist_test(net, testing_dataset)
            validation_performance_over_time.append(validation_performance)

            print("  Testing on training set (this may be slow)...")
            training_performance, _ = mnist_test(net, full_training_dataset)
            training_performance_over_time.append(training_performance)
            
            # The "best" network is determined by its performance on the unseen validation data
            if validation_performance > best_validation_performance:
                print(f"  New best VALIDATION performance found: {validation_performance:.2f}% (at epoch {epoch})")
                best_validation_performance = validation_performance
                best_epoch = epoch
                best_net_state = copy.deepcopy(net)
                best_result_tuple = (best_net_state, validation_performance, incorrect_records)

        # --- CHANGE 3: Plot both lines on the same graph ---
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_numbers, training_performance_over_time, marker='o', linestyle='--', label='Training Performance')
        plt.plot(epoch_numbers, validation_performance_over_time, marker='o', linestyle='--', label='Validation Performance')
        plt.title(f'Performance Over Epochs (Nodes={FIXED_HIDDEN_NODES}, LR={FIXED_LEARNING_RATE},batch_size={FIXED_BATCH_SIZE})')
        plt.xlabel("Epoch")
        plt.ylabel('Performance (%)')
        plt.grid(True)
        # plt.xticks(epoch_numbers)
        plt.legend() # Add a legend to identify the lines
        plt.savefig(os.path.join(save_path, 'train_vs_validation_performance.png'))
        print(f"\nSaved epoch progression plot to {save_path}")
        # plt.close()

        # Analyze and save the network that had the best VALIDATION performance
        if best_result_tuple:
            analyse_best_network(best_result_tuple, fashion_map, save_path)
        
        if best_net_state and input(f"\nSave best trained network (from epoch {best_epoch})? (y/n): ").lower() == 'y':
            net_to_save = best_result_tuple[0]
            filename = f"net_{net_to_save.hidden_nodes}h_{net_to_save.lr}lr_best_at_{best_epoch}e_{net_to_save.batch_size}b.pkl"
            with open(os.path.join(save_path, filename), 'wb') as outp:
                pickle.dump(best_net_state, outp, pickle.HIGHEST_PROTOCOL)
            print(f"Saved network to {os.path.join(save_path, filename)}")
    else:
        print(f"Error: Experiment '{EXPERIMENT_TO_RUN}' not recognized.")

plt.show()