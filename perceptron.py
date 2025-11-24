# Import the NumPy library for matrix math
import numpy as np
import time
import matplotlib.pyplot as plt
# A single perceptron function
def perceptron(inputs_list, weights_list, bias):
    # Convert the inputs list into a numpy array
    inputs = np.array(inputs_list)
    # Convert the weights list into a np array
    weights = np.array(weights_list)
    # Calculate the dot product
    summed = np.dot(inputs, weights)
    # Add in the bias
    biased = summed + bias
    # Calculate output
    # N.B this is a ternary operator, neat huh?
    # print(f"summed {summed} biased {biased}")
    output = 1 if biased > 0 else 0
    return output

# Hidden layer perceptrons (OR and NAND)
def hidden_layer(inputs):
    x1, x2 = inputs
    h1 = perceptron([x1, x2], [1, 1], -0.5)   # OR
    h2 = perceptron([x1, x2], [-1, -1], 1.5)  # NAND
    return h1, h2

# Output perceptron (AND)
def xor_perceptron(inputs):
    h1, h2 = hidden_layer(inputs)
    y = perceptron([h1, h2], [1, 1], -1.5)
    return y

def plot_separator(weights, bias, colour):
    w1, w2 = weights
    x_vals = np.linspace(-0.5, 1.5, 50)
    y_vals = -(w1/w2)*x_vals - (bias/w2)
    plt.plot(x_vals, y_vals, "--", color=colour, label=f"x2= {-w1/w2} * x1 +{-bias/w2}")

def test_perceptron(weights,bias,name):
    # Our main code starts here
    outputs=[]
    print("\nWeights: ", weights)
    print("Bias: ", bias)
    

    for inputs in [[0,0],[0,1],[1,0],[1,1]]:
        output=perceptron(inputs, weights, bias)
        outputs.append(output)
        print(f"In {inputs} Out {output}")
        in1,in2=inputs
        plt.scatter(in1,in2,s=50,color=("green" if output else "red"),zorder=3)
    
    lim1=-0.5
    lim2=1.5

    # plot linear separator
    w1,w2=weights
    x_val=np.linspace(lim1,lim2,8)
    y_val= -(w1/w2) * x_val - (bias/w2)
    plot_separator(weights, bias, "blue")
    
    # plt.plot(x_val,y_val,"--",label=f"x2= {-w1/w2} * x1 +{-bias/w2}")

    plt.xlim(lim1,lim2)
    plt.ylim(lim1,lim2)
    plt.xlabel("x1",fontsize=12)
    plt.ylabel("x2",fontsize=12)
    plt.title(f"{name}: w={weights}, b={bias}")
    plt.xticks(np.linspace(0,1,3))
    plt.yticks(np.linspace(0,1,3))
    plt.grid(True,linewidth=1,linestyle=':')
    plt.legend()
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=10)

def test_xor(name="XOR"):
    # Plot XOR outputs
    for inputs in [[0,0],[0,1],[1,0],[1,1]]:
        output = xor_perceptron(inputs)
        in1, in2 = inputs
        plt.scatter(in1, in2, s=50, color=("green" if output else "red"), zorder=3)
        # plt.text(in1, in2+0.1, f"{output}", ha="center")

    # Plot the linear separators for the *hidden layer* in input space
    
    plot_separator([1, 1], -0.5, "blue")
    plot_separator([-1, -1], 1.5, "orange")

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel("x1",fontsize=12)
    plt.ylabel("x2",fontsize=12)
    plt.title(f"{name}: 2-layers")
    plt.xticks(np.linspace(0,1,3))
    plt.yticks(np.linspace(0,1,3))
    plt.grid(True, linewidth=1, linestyle=':')
    plt.legend()
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=10)


# fig=plt.figure("State Space Analysis")
fig, axs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
plt.sca(axs[0])
test_perceptron([ 1.0, 1.0],-1.5,"AND")
plt.sca(axs[1])
test_perceptron([ 1.0, 1.0],-0.5,"OR")
plt.sca(axs[2])
test_perceptron([-1.0,-1.0], 1.5,"NAND")
plt.sca(axs[3])
test_xor()

for ax in axs:
    ax.xaxis.set_label_coords(0.95, -0.01)
    ax.yaxis.set_label_coords(-0.0, 0.95)
    
# fig = plt.figure("State Space Analysis", figsize=(12, 3))
# plt.subplots_adjust(wspace=0.4)

# plt.tight_layout(pad=2.5)

# plt.tight_layout()
plt.show()