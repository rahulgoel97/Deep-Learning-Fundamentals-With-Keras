# Lab 4 model - attempt to write from scratch with minimal reference.
# Info needed - dataset. 

print("Loading libraries...")
# Libraries
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# CNNs modules
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten


# Get the data
from keras.datasets import mnist
print("Done.")
# Explore MNIST data
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\n\nShape of x-train: {X_train.shape}") #60000, 28, 28

##########################
##########################
##########################
#### SHAPING THE DATA ####
##########################
##########################
##########################


# To shape the data, I need 28x28x3, assuming color image. Is this color?
#print(f"\n\nFirst element on x-train shape{X_train[0].shape} and data: {X_train[0]}")

# Shape should be # of sampels, height, width, num -> as we go to right in matrix, getting closer to
# less abstract value... on the left is more high level

# Number of samples is X_train.shape[0] or 60K

print(f"X_train example before: {X_train[0][0]}")

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

print(f"X_train example after reshape: {X_train[0][0]}")

# Normalize pixel values using 255 which is range of values [0,255] -> zero to 1 map
X_train = X_train / 255 # normalize training data
X_test = X_test / 255 # normalize test data

print(f"X_train example after reshape & normalization: {X_train[0][0]}")

# Turn to categorical -> maps "5" to a vector.
print(f"\n\n\ny_train example before categorical: {y_train[0]}")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"y_train example after categorical: {y_train[0]}")
print(f"y_train shape: {y_test.shape}")
num_classes = y_test.shape[1]


##########################
##########################
##########################
### BUILDING CNN MODEL ###
##########################
##########################
##########################

def convolutional_model():

	model = Sequential()

	# Convolutional and Maxpooling layers
	model.add(Conv2D(16, (5,5), strides=(1,1),activation='relu', input_shape=(28,28,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	# Deep neural net
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	# Compile
	model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

	return model

##########################
##########################
##########################
### DEPLOY CNN MODEL  ###
##########################
##########################
##########################

model = convolutional_model()
print("\n\nTraining CNN model...")
model.fit(X_train, y_train, 
		validation_data=(X_test, y_test),
		epochs=10,
		batch_size=200,
		verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)

print("\nDone.")

print(f"Accuracy: {scores[1]}, Error: {100-scores[1]*100}")