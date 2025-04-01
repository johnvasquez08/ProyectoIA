from ultralytics import YOLO
import os
import torch

# Configuración
DATASET_PATH = "C:/Users/USER/Downloads/ProyectoIA/Oriented.v1i.yolov8"  # Cambia esto con la ruta de tu dataset
MODEL_SAVE_PATH = "C:/Users/USER/Downloads/ProyectoIA/Nuevo modelo"

# Entrenar el modelo YOLOv8 con OBB (Oriented Bounding Boxes)
model = YOLO("yolov8n-obb.pt")  # Modelo base compatible con OBB
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():  # Para Mac con Apple Silicon
    device = "mps"
elif torch.version.hip and torch.cuda.is_available():  # Para AMD con ROCm
    device = "rocm"
else:
    device = "cpu"

model.train(
    data=os.path.join(DATASET_PATH, "data.yaml"),  # Configuración del dataset
    epochs=50,  # Número de épocas de entrenamiento
    imgsz=512,  # Tamaño de la imagen reducido para optimización
    batch=32,  # Aumentar batch si la GPU lo permite
    device=device  # Asignar el dispositivo detectado
)

# Guardar el modelo entrenado
model.export(format="pt", path=MODEL_SAVE_PATH)
print(f"Modelo guardado en {MODEL_SAVE_PATH}")
