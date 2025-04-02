from ultralytics import YOLO
import cv2
import numpy as np

# Cargar el modelo entrenado en YOLOv8-OBB
model = YOLO("runs/obb/train/weights/best.pt")

# Datos de calibración
camera_matrix = np.array([
    [653.04080082, 0.0, 361.37645218],
    [0.0, 873.17495949, 243.84850055],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist_coeffs = np.array(
    [[0.20966857, -0.27011417, -0.02140793, -0.00045373, 0.16516693]], 
    dtype=np.float32)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo conectar a la cámara.")
    exit()

# Variables para almacenar mapas de corrección de distorsión
mapx = None
mapy = None

# Bucle para procesar cada frame del video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    # Obtener dimensiones del frame
    height, width = frame.shape[:2]
    
    # Crear mapas de corrección de distorsión si aún no existen
    if mapx is None or mapy is None:
        # Obtener nuevas matrices de cámara para imágenes sin distorsión
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (width, height), 1, (width, height))
        
        # Calcular mapas de corrección de distorsión
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, 
            (width, height), cv2.CV_32FC1)
    
    # Aplicar corrección de distorsión
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    
    # Redimensionar el frame a 640x640 para la detección
    resized_frame = cv2.resize(undistorted_frame, (640, 640))

    # Realizar la detección en el frame redimensionado y corregido
    results = model.predict(source=resized_frame, conf=0.5)

    # Obtener el frame anotado (con bounding boxes y etiquetas)
    annotated_frame = results[0].plot()

    # Verificar si hay detecciones
    if results[0].obb is not None and results[0].obb.data is not None:
        detections = results[0].obb.data  # Obtener las cajas orientadas
        
        for detection in detections:
            # YOLOv8-OBB devuelve los datos en el formato: (cx, cy, w, h, θ, conf, class)
            cx, cy, w, h, theta = detection[:5].tolist()

            # Dibujar un círculo en el centroide detectado
            cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)  # Verde

            # Mostrar coordenadas del centroide en la imagen
            cv2.putText(annotated_frame, f"({int(cx)}, {int(cy)})", 
                        (int(cx) + 10, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Condición de ubicación en el eje X (ajustar si es necesario)
            if 215 >= cy <= 465:
                print("Posición del objeto en Y:", cy)

    # Agregar indicador de que la imagen está corregida
    cv2.putText(annotated_frame, "Imagen corregida", (annotated_frame.shape[1] - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar el frame con anotaciones
    cv2.imshow("Detección con cámara calibrada", annotated_frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()