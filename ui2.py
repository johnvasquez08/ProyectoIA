import threading
import tkinter as tk
from tkinter import messagebox
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove
from time import sleep
import cv2
import numpy as np
from ultralytics import YOLO

# Variables globales del robot
dashboard, move, feed = None, None, None
robot_connected = False
vision_active = False
model = None
cap = None
Enable = True
velocidad = 50

# Posiciones predefinidas
posiciones = {
    "default": [248, 40, 0, 0],
    "A": [265, 150, -132, 0],
    "B": [265, -150, -132, 0],
}

# Función para loguear mensajes
def log(message):
    status_label.config(text=message)

# Conectar con el robot
def conectar_robot():
    global dashboard, move, feed, robot_connected
    try:
        ip = "192.168.0.11"
        dashboard = DobotApiDashboard(ip, 29999)
        move = DobotApiMove(ip, 30003)
        feed = DobotApi(ip, 30004)
        robot_connected = True
        log("✔ Robot conectado")
    except Exception as e:
        log(f"❌ Error: {str(e)}")
        robot_connected = False

# Activar/desactivar el robot
def activar_robot():
    global Enable
    if not robot_connected:
        log("❌ Robot no conectado")
        return

    try:
        if Enable:
            dashboard.EnableRobot()
            log("✔ Robot activado")
            Enable = False
        else:
            dashboard.DisableRobot()
            log("✔ Robot desactivado")
            Enable = True
    except Exception as e:
        log(f"❌ Error: {str(e)}")

# Mover el robot a una posición
def mover_robot(pos):
    if not robot_connected:
        log("❌ Robot no conectado")
        return

    try:
        move.MovL(*posiciones[pos])
        log(f"✔ Moviendo a posición {pos}")
    except Exception as e:
        log(f"❌ Error: {str(e)}")

# Cambiar la velocidad del robot
def cambiar_velocidad():
    global velocidad
    if not robot_connected:
        log("❌ Robot no conectado")
        return

    try:
        nueva_vel = int(entry_velocidad.get())
        if 1 <= nueva_vel <= 100:
            velocidad = nueva_vel
            dashboard.SpeedL(velocidad)
            log(f"✔ Velocidad ajustada a {velocidad}%")
        else:
            messagebox.showerror("Error", "Velocidad fuera de rango (1-100)")
    except ValueError:
        messagebox.showerror("Error", "Ingrese un número válido")

# Inicializar visión
def iniciar_vision():
    global vision_active, model, cap
    try:
        model = YOLO("runs/detect/train/weights/best.pt")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log("❌ No se pudo abrir la cámara")
            return

        vision_active = True
        log("✔ Visión iniciada")

        def vision_thread():
            while vision_active:
                ret, frame = cap.read()
                if not ret:
                    log("❌ Error en la cámara")
                    break
                results = model.predict(source=frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                cv2.imshow("Visión", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=vision_thread, daemon=True).start()

    except Exception as e:
        log(f"❌ Error en visión: {str(e)}")

# Detener visión
def detener_vision():
    global vision_active
    vision_active = False
    log("✔ Visión detenida")

# Limpiar errores
def limpiar_errores():
    if not robot_connected:
        log("❌ Robot no conectado")
        return
    try:
        dashboard.ClearError()
        log("✔ Errores limpiados")
    except Exception as e:
        log(f"❌ Error: {str(e)}")

# Crear la ventana principal
root = tk.Tk()
root.title("Control de Robot y Visión")
root.geometry("400x450")
root.resizable(False, False)

# Etiqueta de estado
status_label = tk.Label(root, text="Estado: Desconectado", font=("Arial", 12), fg="blue")
status_label.pack(pady=10)

# Botones de control
btn_conectar = tk.Button(root, text="Conectar Robot", command=conectar_robot, font=("Arial", 12))
btn_conectar.pack(pady=5)

btn_activar = tk.Button(root, text="Activar/Desactivar Robot", command=activar_robot, font=("Arial", 12))
btn_activar.pack(pady=5)

btn_mover_A = tk.Button(root, text="Mover a Posición A", command=lambda: mover_robot("A"), font=("Arial", 12))
btn_mover_A.pack(pady=5)

btn_mover_B = tk.Button(root, text="Mover a Posición B", command=lambda: mover_robot("B"), font=("Arial", 12))
btn_mover_B.pack(pady=5)

# Controles de velocidad
tk.Label(root, text="Velocidad (1-100):", font=("Arial", 12)).pack(pady=5)
entry_velocidad = tk.Entry(root, font=("Arial", 12), justify="center")
entry_velocidad.pack(pady=5)
btn_velocidad = tk.Button(root, text="Ajustar Velocidad", command=cambiar_velocidad, font=("Arial", 12))
btn_velocidad.pack(pady=5)

# Controles de visión
btn_iniciar_vision = tk.Button(root, text="Iniciar Visión", command=iniciar_vision, font=("Arial", 12))
btn_iniciar_vision.pack(pady=5)

btn_detener_vision = tk.Button(root, text="Detener Visión", command=detener_vision, font=("Arial", 12))
btn_detener_vision.pack(pady=5)

# Limpiar errores
btn_limpiar = tk.Button(root, text="Limpiar Errores", command=limpiar_errores, font=("Arial", 12))
btn_limpiar.pack(pady=5)

# Iniciar la interfaz
root.mainloop()
