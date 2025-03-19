from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("./runs/detect/train/weights/best.pt")

# Hacer una predicción en una imagen
results = model.predict(source="C:/Users/USER/Downloads/ProyectoIA\Proyectoprimeraprueba.v1i.yolov8/test/images/unknown_5mhku1mf_ingestion-5fc558967f-tlxdc_jpg.rf.9b89051d80892ece89e7d8c114a1c44a.jpg", save=True)