import pickle
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import hickle

CATEGORIES = ("cats", "dogs")
RES_X = 128
RES_Y = 128

X = []
Y = []


def build_output(index):
    out = [0 for _ in CATEGORIES]
    out[index] = 1
    return out


if __name__ == "__main__":
    # get listdir from every directory => all files for every category
    category_paths = [
        os.listdir(os.path.join(os.getcwd(), f"dataset/{category}"))
        for category in CATEGORIES
        ]

    # iterate through each samples
    for i in tqdm(range(len(category_paths[0]))):
        for k, paths in enumerate(category_paths):
            try:
                image_full_path = f"{os.getcwd()}/dataset/{CATEGORIES[k]}/{paths[i]}"

                img = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
                resized = cv2.resize(img, (RES_X, RES_Y))

                # plt.imshow(img)
                # plt.show()

                X.append(resized)
                Y.append(build_output(k))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)

    X = np.array(X).reshape((-1, RES_X, RES_Y, 3))
    Y = np.array(Y)

    X = X / 255

    # pickle.dump(X, open("pickles/X.pickle", "wb"), protocol=4)
    # pickle.dump(Y, open("pickles/Y.pickle", "wb"), protocol=4)
    hickle.dump(X, "pickles/X.hkl")
    hickle.dump(Y, "pickles/Y.hkl")

