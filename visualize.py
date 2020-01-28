import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV2
import cv2
from preprocessing import RES_Y, RES_X
import numpy as np
import os

def preprocess_input(x):
    x /= 255
    x -= 0.5
    x *= 2
    return x


np.seterr(divide='ignore', invalid='ignore')
# model = load_model("model/main.h5")

model = MobileNetV2()
model.summary()
print(len(model.layers))

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

img = cv2.imread("test/dog2.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = np.expand_dims(np.array(img), axis=0) / 255
img = preprocess_input(img)

activations = activation_model.predict(img)

layer_names = []
for layer in model.layers[120:130]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,  # Displays the grid
            row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()
