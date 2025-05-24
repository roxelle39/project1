# -*- coding: utf-8 -*-
import tensorflow as tf
from tf_data_prep import get_datasets
from tf_model import create_model

def train():
    train_ds, val_ds = get_datasets("breast_cancer")
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save("rokhaya_model.tensorflow")
    print("✅ TensorFlow model saved.")

if __name__ == "__main__":
    train()
