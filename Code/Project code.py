# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 11:07:02 2025

@author: jcous
"""
# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2

# Cargar datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train_small, y_train_small = x_train[:2000], y_train[:2000]
x_test_small, y_test_small = x_test[:500], y_test[:500]

# Mostrar un ejemplo por clase
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(10, 5))
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("class_examples.png")
plt.close()

# Función para extraer características manuales
def extract_features(images):
    features = []
    for img in images:
        mean = np.mean(img)
        std = np.std(img)
        nonzero = np.count_nonzero(img)
        edges = cv2.Canny(img, 100, 200)
        edge_count = np.count_nonzero(edges)
        features.append([mean, std, nonzero, edge_count])
    return np.array(features)

# Extraer y normalizar características
x_train_manual = extract_features(x_train_small)
x_test_manual = extract_features(x_test_small)
scaler = StandardScaler()
x_train_manual = scaler.fit_transform(x_train_manual)
x_test_manual = scaler.transform(x_test_manual)

# Modelo manual (k-NN)
manual_model = KNeighborsClassifier(n_neighbors=5)
manual_model.fit(x_train_manual, y_train_small)
y_pred_manual = manual_model.predict(x_test_manual)

# Modelo automático (CNN)
x_train_cnn = x_train_small.reshape(-1, 28, 28, 1) / 255.0
x_test_cnn = x_test_small.reshape(-1, 28, 28, 1) / 255.0

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(x_train_cnn, y_train_small, validation_split=0.2, epochs=5)

# Predicciones CNN
y_pred_cnn = np.argmax(cnn_model.predict(x_test_cnn), axis=1)

# Matriz de confusión manual
cm_manual = confusion_matrix(y_test_small, y_pred_manual)
ConfusionMatrixDisplay(cm_manual).plot(cmap='Blues')
plt.title("Manual Features - Confusion Matrix")
plt.savefig("confusion_matrix_manual.png")
plt.close()

# Matriz de confusión CNN
cm_cnn = confusion_matrix(y_test_small, y_pred_cnn)
ConfusionMatrixDisplay(cm_cnn).plot(cmap='Greens')
plt.title("CNN - Confusion Matrix")
plt.savefig("confusion_matrix_cnn.png")
plt.close()

# Accuracy vs Epoch (CNN)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("cnn_accuracy_epochs.png")
plt.close()



# Gráfico 1: Distribución de clases
plt.figure(figsize=(8, 5))
counts = [np.sum(y_train == i) for i in range(10)]
plt.bar(class_names, counts, color='skyblue')
plt.title("Class Distribution in Training Set")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# Gráfico 2: Histograma de intensidades de píxeles
plt.figure(figsize=(8, 5))
all_pixels = x_train.flatten()
plt.hist(all_pixels, bins=30, color='salmon', edgecolor='black')
plt.title("Pixel Intensity Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("pixel_intensity_histogram.png")
plt.show()

# Manual features
def extract_manual_features(images):
    features = []
    for img in images:
        mean = np.mean(img)
        std = np.std(img)
        nonzero = np.count_nonzero(img)
        edges = cv2.Canny(img, 100, 200)
        edge_count = np.count_nonzero(edges)
        features.append([mean, std, nonzero, edge_count])
    return np.array(features)

manual_train = extract_manual_features(x_train_small)
manual_test = extract_manual_features(x_test_small)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(manual_train, y_train_small)
knn_preds = knn.predict(manual_test)

# CNN
x_train_cnn = x_train_small / 255.0
x_test_cnn = x_test_small / 255.0
x_train_cnn = x_train_cnn.reshape(-1, 28, 28, 1)
x_test_cnn = x_test_cnn.reshape(-1, 28, 28, 1)

cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train_cnn, y_train_small, epochs=5, verbose=0)
cnn_preds = np.argmax(cnn.predict(x_test_cnn), axis=1)

# Visual error comparison
def get_errors(preds, true_labels, images, max_errors=3):
    errors = []
    for i in range(len(preds)):
        if preds[i] != true_labels[i]:
            errors.append((images[i], true_labels[i], preds[i]))
        if len(errors) == max_errors:
            break
    return errors

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

knn_errors = get_errors(knn_preds, y_test_small, x_test_small)
cnn_errors = get_errors(cnn_preds, y_test_small, x_test_small)

# Plot
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
for i in range(3):
    img, true, pred = knn_errors[i]
    axs[0, i].imshow(img, cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title(f'True: {class_names[true]}\nPred: {class_names[pred]}')

    img, true, pred = cnn_errors[i]
    axs[1, i].imshow(img, cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title(f'True: {class_names[true]}\nPred: {class_names[pred]}')

plt.tight_layout()
plt.savefig("error_visualization.png")
plt.show()

