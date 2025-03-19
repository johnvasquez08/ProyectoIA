from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)

# Cargar el modelo YOLO
model = YOLO("runs/detect/train/weights/best.pt")

# URL de la cámara IP
camera_ip = "http://192.168.80.32:8080/video"  # Cambia por la dirección IP de tu teléfono

# Variables globales
cap = None
last_detection_time = 0
capture_image = None
centroid_coords = None

# Función para generar frames de video
def generate_frames():
    global last_detection_time, capture_image, centroid_coords

    cap = cv2.VideoCapture(camera_ip)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame a 640x640
        resized_frame = cv2.resize(frame, (640, 640))

        # Realizar la detección en el frame redimensionado
        results = model.predict(source=resized_frame, conf=0.5)

        # Obtener el frame anotado (con bounding boxes y etiquetas)
        annotated_frame = results[0].plot()

        # Obtener las detecciones
        detections = results[0].boxes.data

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

            # Guardar las coordenadas del centroide
            centroid_coords = (centroid_x, centroid_y)

        # Verificar si hay detecciones
        if len(detections) > 0:
            # Si es la primera detección o han pasado más de 2 segundos desde la última
            if time.time() - last_detection_time >= 2:
                # Tomar una captura de pantalla del frame anotado
                capture_image = annotated_frame.copy()
                last_detection_time = time.time()

        # Convertir el frame anotado a formato JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Enviar el frame al cliente
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para el video en tiempo real
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para obtener la captura de pantalla
@app.route('/capture')
def capture():
    global capture_image
    if capture_image is not None:
        ret, buffer = cv2.imencode('.jpg', capture_image)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "No capture available", 404

# Ruta para obtener las coordenadas del centroide
@app.route('/centroid')
def centroid():
    global centroid_coords
    if centroid_coords is not None:
        return jsonify({"x": centroid_coords[0], "y": centroid_coords[1]})
    return jsonify({"x": None, "y": None})

# Iniciar el servidor Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)