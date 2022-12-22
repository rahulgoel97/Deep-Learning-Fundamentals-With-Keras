# Neural network

'''
Requirements

- Inputs: number of inputs
- Layers: number of hidden layers
- Nodes: number of nodes per layer
- Output: 1
'''

n = 2
num_hidden_layers = 2
m = [2,2]
num_nodes_output = 1

'''
Structure


- Initialize network
- Compute weighted sums at a node
- Activation function
- Forward Prop

'''

import numpy as np



def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
	
	network = {}
	num_nodes_previous = num_inputs

	for layer in range(num_hidden_layers+1):

		# Determine the name of the layer
		if(layer==num_hidden_layers):
			layer_name='output'
			num_nodes = num_nodes_output
		else:
			layer_name = f'layer {layer+1}'
			num_nodes = num_nodes_hidden[layer]

		# Intiialise weights and biases
		network[layer_name] = {}

		for node in range(num_nodes):
			node_name = f'node {node+1}'
			network[layer_name][node_name] = {
			'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
			'bias': np.around(np.random.uniform(size=1), decimals=2),
			}
		num_nodes_previous = num_nodes

	return network


# Compute weighted sums
def compute_weighted_sums(inputs, weights, bias):
	return np.sum(inputs*weights) + bias

# Activation - sigmoid
def node_activation(weighted_sum):
	return 1.0/(1.0 + np.exp(-1 * weighted_sum))

# Forward prop
def forward_propagate(network, inputs):

	layer_inputs = list(inputs)

	for layer in network:

		layer_data = network[layer]

		layer_outputs = []

		for layer_node in layer_data:

			node_data = layer_data[layer_node]

			node_output = node_activation(compute_weighted_sums(layer_inputs, node_data['weights'], node_data['bias']))

			layer_outputs.append(np.around(node_output[0], decimals=4))

		

		layer_inputs = layer_outputs

		network_predictions = layer_outputs

		return network_predictions 

# Start the nn

my_network = initialize_network(5, 3, [2,3,2], 3)
inputs = np.around(np.random.uniform(size=5), decimals=2)
predictions = forward_propagate(my_network, inputs)

print("####### Network #######", my_network)
print("\n\n###### Inputs ######", inputs)
print("\n\n###### Predictions ######", predictions)