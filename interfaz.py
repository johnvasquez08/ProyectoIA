from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import numpy as np

# Cargar el modelo entrenado
model = YOLO("C:/Users/johnd/Documents/proyectofinal/ProyectoIA/runs/detect/train/weights/best.pt")

# URL de la cámara IP
camera_ip = "http://172.20.10.6:8080/video"

# Variables globales
cap = None
is_camera_running = False
object_positions = {}  # Diccionario para guardar posiciones anteriores
fps = 30  # Ajusta según el FPS real de la cámara
PIXELS_PER_METER = 100  # Ajusta este valor según la escala real

# Función para iniciar la captura de video
def start_camera():
    global cap, is_camera_running
    if not is_camera_running:
        cap = cv2.VideoCapture(camera_ip)
        is_camera_running = True
        show_frame()

# Función para detener la captura de video
def stop_camera():
    global cap, is_camera_running
    if is_camera_running:
        cap.release()
        is_camera_running = False
        video_label.config(image='')

# Función para mostrar el video en la interfaz
def show_frame():
    global object_positions

    if is_camera_running:
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (640, 640))
            results = model.predict(source=resized_frame, conf=0.5)
            annotated_frame = results[0].plot()
            detections = results[0].boxes.data

            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection[:4].int().tolist()
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                object_id = i  # Asignar un ID temporal basado en el índice de detección

                # Si ya tenemos la posición anterior del objeto
                if object_id in object_positions:
                    prev_x, prev_y, prev_time = object_positions[object_id]
                    distance_px = np.sqrt((centroid_x - prev_x) ** 2 + (centroid_y - prev_y) ** 2)
                    time_diff = time.time() - prev_time

                    if time_diff > 0:
                        # Convertir de píxeles a metros
                        distance_m = distance_px / PIXELS_PER_METER
                        speed_m_per_s = distance_m / time_diff

                        # Mostrar la velocidad en m/s en la imagen
                        cv2.putText(annotated_frame, f"{speed_m_per_s:.2f} m/s", 
                                    (centroid_x + 10, centroid_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Guardar la posición actual del objeto para el siguiente frame
                object_positions[object_id] = (centroid_x, centroid_y, time.time())

                # Dibujar círculo en el centroide
                cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

            # Convertir el frame anotado a formato Tkinter
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

        video_label.after(10, show_frame)

# Función para cerrar la aplicación
def close_app():
    if is_camera_running:
        cap.release()
    root.destroy()

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Detección y Velocidad en m/s con YOLO")
root.geometry("1300x700")

video_frame = ttk.Frame(root)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)
video_label = ttk.Label(video_frame)
video_label.pack()

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)
ttk.Button(button_frame, text="Iniciar cámara", command=start_camera).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Detener cámara", command=stop_camera).pack(side=tk.LEFT, padx=5)
ttk.Button(button_frame, text="Cerrar", command=close_app).pack(side=tk.LEFT, padx=5)

root.mainloop()
