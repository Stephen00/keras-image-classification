import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale image data into decimal value
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    # The probability for each given class
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Epochs gives the same images in a different order, in effort to increase the accuracy of our model
model.fit(train_images, train_labels, epochs=10)

# Create our probability model and make predictions

predictions = model.predict(test_images)

for i in range(10):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    predicted_label = class_names[np.argmax(predictions[i])]
    if predicted_label == class_names[test_labels[i]]:
        colour = 'blue'
    else:
        colour = 'red'
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]], color=colour)
    plt.ylabel("Prediction: " + class_names[np.argmax(predictions[i])], color='blue')
    plt.show()



