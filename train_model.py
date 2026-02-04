import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
IMG_DIR = os.path.join(DATASET_DIR, "images")
MASK_DIR = os.path.join(DATASET_DIR, "masks")
CSV_PATH = os.path.join(DATASET_DIR, "mapping.csv")

IMG_SIZE = 256
EPOCHS = 5
BATCH_SIZE = 8

# ---------------- LOAD CSV ----------------
df = pd.read_csv(CSV_PATH)
print("Total samples:", len(df))

# ---------------- IMAGE LOADER ----------------
def load_image_pair(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)

    return img, mask

import random

def data_generator(df, batch_size=2):
    while True:
        df = df.sample(frac=1)

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]

            X_batch, Y_batch = [], []

            for _, row in batch.iterrows():
                img_path = os.path.join(IMG_DIR, row["image"])
                mask_path = os.path.join(MASK_DIR, row["mask"])

                img, mask = load_image_pair(img_path, mask_path)
                X_batch.append(img)
                Y_batch.append(mask)

            yield np.array(X_batch), np.array(Y_batch)





# ---------------- BUILD SIMPLE U-NET ----------------
def build_unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation="relu", padding="same")(p1)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation="relu", padding="same")(p2)

    u4 = UpSampling2D()(c3)
    c4 = Conv2D(64, 3, activation="relu", padding="same")(u4)

    u5 = UpSampling2D()(c4)
    c5 = Conv2D(32, 3, activation="relu", padding="same")(u5)

    outputs = Conv2D(1, 1, activation="sigmoid")(c5)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

train_df, val_df = train_test_split(df, test_size=0.2)

train_gen = data_generator(train_df, batch_size=2)
val_gen = data_generator(val_df, batch_size=2)


# ---------------- TRAIN MODEL ----------------
model = build_unet()
model.summary()

model.fit(
    train_gen,
    steps_per_epoch=len(train_df)//2,
    validation_data=val_gen,
    validation_steps=len(val_df)//2,
    epochs=3
)




# ---------------- SAVE MODEL ----------------
model.save("forest_segmentation_model.h5")
print("Model saved successfully")

