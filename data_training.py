import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialize variables and lists for data loading and preprocessing
is_init = False
size = -1

# Define the two emotions you want to classify
emotions = ["happy", "sad"]

# Create a dictionary to map emotions to integers
emotion_to_int = {emotion: i for i, emotion in enumerate(emotions)}

X = None
y = None

# Load the data for the specified emotions
for emotion in emotions:
    file_name = f"{emotion}.npy"
    if os.path.exists(file_name):
        data = np.load(file_name)
        if X is None:
            X = data
        else:
            X = np.concatenate((X, data))
        labels = np.array([emotion_to_int[emotion]] * data.shape[0]).reshape(-1, 1)
        if y is None:
            y = labels
        else:
            y = np.concatenate((y, labels))

# Convert class labels to categorical one-hot encoding
y = to_categorical(y, num_classes=len(emotions))

# Shuffle the data and labels randomly
random_indices = np.arange(X.shape[0])
np.random.shuffle(random_indices)
X = X[random_indices]
y = y[random_indices]

# Define the neural network model
input_layer = Input(shape=(X.shape[1]))
hidden_layer_1 = Dense(512, activation="relu")(input_layer)
hidden_layer_2 = Dense(256, activation="relu")(hidden_layer_1)
output_layer = Dense(len(emotions), activation="softmax")(hidden_layer_2)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model using categorical cross-entropy loss and the RMSprop optimizer
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50)

# Save the trained model to a file
model.save("emotion_model.h5")

# Save the list of emotions to a file
np.save("emotions.npy", np.array(emotions))
