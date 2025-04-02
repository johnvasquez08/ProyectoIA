from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("C:/Users/johnd/Documents/proyectofinal/ProyectoIA/runs/obb/train/weights/best.pt")

# Hacer una predicción en una imag/n
results = model.predict(source="C:/Users/johnd/Documents/proyectofinal/ProyectoIA/Fotos/foto_20250328_114538_2.jpg", save=True)