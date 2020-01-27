from tensorflow_core.python.keras.models import load_model
import numpy as np
import cv2
from preprocessing import RES_X, RES_Y
import matplotlib.pyplot as plt


model = load_model("model/main.h5")
img = cv2.imread("test/dog.jpeg", cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (RES_X, RES_Y))
plt.imshow(img)
plt.show()
model.summary()
img = np.expand_dims(np.array(img), axis=0) / 255

out = model.predict(img)[0]
print(out)