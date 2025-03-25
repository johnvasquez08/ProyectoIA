from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("runs/detect/train/weights/best.pt")  # Reemplaza con la ruta a tu best.pt

# URL de la cámara IP (cambia por la dirección IP de tu teléfono)
camera_ip = "http://192.168.80.32:8080/video"  # Para IP Webcam
# camera_ip = "http://192.168.1.100:4747/video"  # Para DroidCam

# Iniciar la captura de video desde la cámara IP
cap = cv2.VideoCapture(3)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo conectar a la cámara IP.")
    exit()

# Bucle para procesar cada frame del video
while True:
    # Capturar un frame
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Redimensionar el frame a 640x640
    resized_frame = cv2.resize(frame, (640, 640))

    # Realizar la detección en el frame redimensionado
    results = model.predict(source=resized_frame, conf=0.5)  # conf es el umbral de confianza

    # Obtener el frame anotado (con bounding boxes y etiquetas)
    annotated_frame = results[0].plot()  # Frame con las bounding boxes dibujadas

    # Obtener las detecciones
    detections = results[0].boxes.data  # Obtener las cajas delimitadoras y las clases

    # Recorrer cada detección
    for detection in detections:
        # Obtener las coordenadas de la caja delimitadora (x1, y1, x2, y2)
        x1, y1, x2, y2 = detection[:4].int().tolist()

        # Calcular el centroide (punto central de la caja delimitadora)
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2

        # Dibujar un círculo en el centroide
        cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)  # Círculo verde

        # (Opcional) Mostrar las coordenadas del centroide
        cv2.putText(annotated_frame, f"({centroid_x}, {centroid_y})", (centroid_x + 10, centroid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar el frame con las bounding boxes y los centroides
    cv2.imshow("Detección en tiempo real (Cámara IP)", annotated_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()