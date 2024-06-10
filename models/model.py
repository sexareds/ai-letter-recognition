import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, utils
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Cargar datos del dataset"""
    try:
        training_data = pd.read_csv('emnist/emnist-letters-train.csv')
        testing_data = pd.read_csv('emnist/emnist-letters-test.csv')
        print("CSV files read successfully.")
    except FileNotFoundError:
        print("Error: CSV files not found in the specified location.")
        return None, None
    return training_data, testing_data

def preprocess_data(data):
    """Preprocess the data: scale, flatten, and one-hot encode."""
    # Subtract 1 from labels to shift from 0-25
    y = utils.to_categorical(data.iloc[:, 0] - 1, num_classes=26)
    x = data.iloc[:, 1:].values / 255.0
    x = x.reshape(-1, 28, 28, 1)
    return x, y

def build_model(num_classes):
    """Build a Sequential neural network model."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_history(history):
    """Plot training and validation accuracy."""
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=history.history['accuracy'], label='Accuracy')
    sns.lineplot(x=epochs, y=history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def main():
    train_data, test_data = load_data()
    
    if train_data is None or test_data is None:
        print("Failed to load data. Exiting...")
        return

    train_x, train_y = preprocess_data(train_data)
    test_x, test_y = preprocess_data(test_data)
    
    model = build_model(26)  # Número actualizado de clases
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    
    history = model.fit(train_x, train_y, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
    plot_history(history)

    # evaluacion del modelo con los datos de prueba
    loss, accuracy = model.evaluate(test_x, test_y)
    print(accuracy, loss)
    
    model.save('models/saved_models/model.keras')

if __name__ == "__main__":
    main()
