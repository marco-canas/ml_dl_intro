### Redes Neuronales con Keras y TensorFlow
# Basado en Géron (3ª edición), págs. 508–513
# Diseño de práctica interactiva con enfoque activo, experimental y reflexivo.

# ============================================================
# CELDA 1 - Introducción (Markdown)
# ============================================================
"""
# Práctica: Redes Neuronales en Keras

En esta práctica exploraremos cómo construir, entrenar y evaluar redes neuronales artificiales con **Keras** y **TensorFlow**, siguiendo lo presentado en Géron (págs. 508–513).

**Objetivos:**
- Comprender la construcción de redes neuronales con la API secuencial de Keras.
- Entrenar un modelo para clasificar imágenes del dataset **Fashion MNIST**.
- Evaluar el desempeño del modelo.
- Reflexionar sobre los hiperparámetros (número de capas, neuronas, función de activación, etc.).
"""

# ============================================================
# CELDA 2 - Importaciones
# ============================================================
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CELDA 3 - Cargar datos
# ============================================================
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalización
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Separar validación
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ============================================================
# CELDA 4 - Visualización de ejemplos
# ============================================================
n = 10
plt.figure(figsize=(10,2))
for i in range(n):
    plt.subplot(1, n, i+1)
    plt.imshow(X_train[i], cmap="binary")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.show()

# ============================================================
# CELDA 5 - Construcción del modelo
# ============================================================
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# ============================================================
# CELDA 6 - Compilación del modelo
# ============================================================
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# ============================================================
# CELDA 7 - Entrenamiento
# ============================================================
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

# ============================================================
# CELDA 8 - Curvas de aprendizaje
# ============================================================
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="valid")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()

# ============================================================
# CELDA 9 - Evaluación en test
# ============================================================
eval_result = model.evaluate(X_test, y_test)
print(f"\nPérdida en test: {eval_result[0]:.4f}")
print(f"Precisión en test: {eval_result[1]:.4f}")

# ============================================================
# CELDA 10 - Predicciones
# ============================================================
y_pred = model.predict(X_test[:3])
print("Predicciones (probabilidades):")
print(y_pred)
print("Clases predichas:", np.argmax(y_pred, axis=1))
print("Clases reales:", y_test[:3])

# ============================================================
# CELDA 11 - Preguntas reflexivas (Markdown)
# ============================================================
"""
## Preguntas para reflexionar:

1. ¿Qué efecto tuvo aumentar/disminuir el número de neuronas en las capas ocultas?
2. ¿Cómo cambia la precisión si reemplazamos la función de activación `relu` por `tanh`?
3. ¿Qué ocurre si reducimos las épocas de entrenamiento a 5 o las aumentamos a 50?
4. ¿Qué diferencia encuentras entre la pérdida en entrenamiento y validación?
5. ¿Cómo podríamos mejorar este modelo? (pistas: optimizadores, regularización, dropout, etc.)
"""
