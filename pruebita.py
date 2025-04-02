import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar datos
data = pd.read_csv('datos.csv')

# Separar características y etiquetas
X = data[['centroid_x', 'centroid_y', 'belt_speed', 'robot_x', 'robot_y', 'robot_z']]
y = data[['target_x', 'target_y', 'target_z']]

# Normalizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(3)  # Salida para x, y, z
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

import joblib
from tensorflow.keras.models import save_model

# Guardar el modelo de Keras
save_model(model, 'robot_picking_model.h5')

# Guardar el scaler para normalización
joblib.dump(scaler, 'scaler.save')

# Opcional: Guardar la arquitectura del modelo como JSON (si necesitas reconstruirlo manualmente)
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)