import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Cargar los datos recolectados (se agrega on_bad_lines='skip' para ignorar filas corruptas)
print("Cargando dataset...")
df = pd.read_csv('datasetLETRAS.csv', on_bad_lines='skip')

# Separar las coordenadas (X) de las respuestas (Y)
X = df.iloc[:, 1:].values # Las 42 columnas de coordenadas
y = df.iloc[:, 0].values  # La columna 0 con el número de la clase

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Crear la red neuronal para coordenadas (es muy rápida y ligera)
print("Creando modelo...")
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(42,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax') # 9 clases en total (0-8)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Entrenar el modelo
print("Entrenando modelo...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# 4. Convertir y guardar en formato TFLite
print("Convirtiendo a TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# SE CORRIGIÓ EL NOMBRE DEL ARCHIVO PARA NO SOBRESCRIBIR EL MODELO PRINCIPAL
with open('modelo_teclado.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Entrenamiento terminado. Archivo 'modelo_teclado.tflite' creado con éxito.")