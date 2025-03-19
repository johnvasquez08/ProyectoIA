from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

# Cargar el modelo entrenado
model = YOLO("runs/detect/train/weights/best.pt")  # Reemplaza con la ruta a tu best.pt

# URL de la cámara IP (cambia por la dirección IP de tu teléfono)
camera_ip = "http://192.168.80.32:8080/video"  # Para IP Webcam
# camera_ip = "http://192.168.1.100:4747/video"  # Para DroidCam

# Variables globales
cap = None
is_camera_running = False
last_detection_time = 0
capture_image = None

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
        video_label.config(image='')  # Limpiar la imagen en la interfaz

# Función para mostrar el video en la interfaz
def show_frame():
    global last_detection_time, capture_image

    if is_camera_running:
        ret, frame = cap.read()
        if ret:
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

            # Verificar si hay detecciones
            if len(detections) > 0:
                # Si es la primera detección o han pasado más de 2 segundos desde la última
                if time.time() - last_detection_time >= 2:
                    # Tomar una captura de pantalla del frame anotado
                    capture_image = annotated_frame.copy()
                    last_detection_time = time.time()

                    # Mostrar la captura al lado del video
                    show_capture(capture_image)

            # Convertir el frame anotado a formato compatible con Tkinter
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Mostrar el frame en la interfaz
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

        # Llamar a la función nuevamente después de 10 ms
        video_label.after(10, show_frame)

# Función para mostrar la captura en la interfaz
def show_capture(capture):
    # Convertir la captura a formato compatible con Tkinter
    capture = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
    capture_img = Image.fromarray(capture)
    capture_imgtk = ImageTk.PhotoImage(image=capture_img)

    # Mostrar la captura en la etiqueta correspondiente
    capture_label.imgtk = capture_imgtk
    capture_label.config(image=capture_imgtk)

# Función para cerrar la aplicación
def close_app():
    if is_camera_running:
        cap.release()
    root.destroy()

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Detección en tiempo real con YOLO")
root.geometry("1300x700")  # Ajustar el tamaño para acomodar el video y la captura

# Variable para controlar el estado de la cámara
is_camera_running = False

# Marco para el video en tiempo real
video_frame = ttk.Frame(root)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Etiqueta para mostrar el video
video_label = ttk.Label(video_frame)
video_label.pack()

# Marco para la captura de pantalla
capture_frame = ttk.Frame(root)
capture_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Etiqueta para mostrar la captura
capture_label = ttk.Label(capture_frame)
capture_label.pack()

# Botones de control
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

start_button = ttk.Button(button_frame, text="Iniciar cámara", command=start_camera)
start_button.pack(side=tk.LEFT, padx=5)

stop_button = ttk.Button(button_frame, text="Detener cámara", command=stop_camera)
stop_button.pack(side=tk.LEFT, padx=5)

close_button = ttk.Button(button_frame, text="Cerrar", command=close_app)
close_button.pack(side=tk.LEFT, padx=5)

# Iniciar el bucle principal de Tkinter
root.mainloop()