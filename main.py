from ultralytics import YOLO

# Cargar un modelo preentrenado (por ejemplo, YOLOv8n)
model = YOLO("yolov8n.pt")

# Entrenar el modelo
results = model.train(data="C:/Users/USER/Downloads/ProyectoIA/Proyectoprimeraprueba.v1i.yolov8/data.yaml", epochs=100, imgsz=640)