import os
import cv2
from tqdm import tqdm
import imgaug.augmenters as iaa
import numpy as np
from preprocessing import CATEGORIES, RES_X, RES_Y

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
AUGMENT_PER_IMAGE = 6
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            rotate=(-20, 20),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            mode="edge"  # use any of scikit-image"s warping modes (see 2nd image from the top for examples)
        )),
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
        iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        sometimes(iaa.OneOf([
            iaa.SigmoidContrast(gain=5),
            iaa.Multiply(mul=1.0),
            iaa.MultiplyHueAndSaturation(mul=1.25)
        ])),
    ],
    random_order=True
)


def apply_batch(images, file_counter_start, aug_amount):
    i = file_counter_start
    batches = []
    for _ in range(aug_amount):
        batches.append(iaa.UnnormalizedBatch(images=images))

    for batch in seq.augment_batches(batches=batches, background=True):
        for images in batch.images_aug:
            cv2.imwrite(f"augmented/{category}/{category}_{i}.png", images)
            i += 1
    return i


if __name__ == "__main__":
    for category in CATEGORIES:
        DIR = os.path.join(os.getcwd(), f"images\\{category}")
        images = list()

        i = 0
        for img_name in tqdm(os.listdir(DIR)):
            try:
                img = cv2.imread(os.path.join(DIR, img_name), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

                if len(images) > 1000:
                    i = apply_batch(images, file_counter_start=i, aug_amount=AUGMENT_PER_IMAGE)
                    images = list()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)
        
        if len(images) > 0:
            apply_batch(images, file_counter_start=i, aug_amount=AUGMENT_PER_IMAGE)
