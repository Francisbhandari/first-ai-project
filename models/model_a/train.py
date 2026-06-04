import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D

EPOCHS = 10
BATCH_SIZE =128
MODEL_PATH = "model1.keras"


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255
    x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training...")
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("Evaluating...")
    loss, acc = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {acc:.4f}\nTest loss: {loss:.4f}")

    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("Done.")
