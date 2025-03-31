import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile
from time import sleep
import numpy as np
import time
import re
import os
import cv2
from ultralytics import YOLO

# Configuración del sistema
VERBOSE = False  # Cambiar a True para más detalles en consola

# Variables globales del robot
current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()
Enable = True
velocidad = 50
robot_connected = False

# Posiciones predefinidas
posiciones = {
    "default": [248, 40, 0, 0],
    "A": [265, 150, -132, 0],
    "B": [265, -150, -132, 0],
    "C": [259, -109, 10, 0],
    "D": [259, 100, 10, 0],
    "E": [259, 100, -136, 0]
}

# Variables de visión
vision_active = False
model = None
cap = None
dashboard, move, feed = None, None, None

def log(message, level="info"):
    """Sistema de logging controlado por verbosidad"""
    if level == "error" or VERBOSE:
        print(message)

def ConnectRobot():
    """Intenta conectar con el robot Dobot"""
    global dashboard, move, feed, robot_connected
    try:
        ip = "192.168.0.11"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        log("Estableciendo conexión con el robot...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        robot_connected = True
        log("> Conexión con robot exitosa <")
        return True
    except Exception as e:
        log(f"× Conexión con robot fallida: {str(e)}", "error")
        robot_connected = False
        return False
def SpeedFactor(dashboard: DobotApiDashboard, speed):
        """
    Setting the Global rate
    speed:Rate value(Value range:1~100)
    """
        string = "SpeedFactor({:d})".format(speed)
        return dashboard.sendRecvMsg(string)
def SpeedL(dashboard: DobotApiDashboard, speed):
        """
    Set the cartesian acceleration ratio (Only for MovL, MovLIO, MovLR, Jump, Arc, Circle commands)
    speed : Cartesian acceleration ratio (Value range:1~100)
    """
        string = "SpeedL({:d})".format(speed)
        return dashboard.sendRecvMsg(string)
def InitializeVision():
    """Inicializa el sistema de visión artificial"""
    global model, cap
    try:
        log("Inicializando sistema de visión...")
        model = YOLO("runs/detect/train/weights/best.pt")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            log("× No se pudo conectar a la cámara", "error")
            return False
            
        log("> Visión artificial inicializada <")
        return True
    except Exception as e:
        log(f"× Error al inicializar visión: {str(e)}", "error")
        return False

def GetFeed(feed: DobotApi):
    """Hilo para obtener feedback del robot (silencioso)"""
    global current_actual, algorithm_queue, enableStatus_robot, robotErrorState
    hasRead = 0
    while True:
        try:
            if feed is None:
                sleep(1)
                continue
                
            data = bytes()
            while hasRead < 1440:
                temp = feed.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            hasRead = 0
            feedInfo = np.frombuffer(data, dtype=MyType)
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                with globalLockValue:
                    current_actual = feedInfo["tool_vector_actual"][0]
                    algorithm_queue = feedInfo['isRunQueuedCmd'][0]
                    enableStatus_robot = feedInfo['EnableStatus'][0]
                    robotErrorState = feedInfo['ErrorStatus'][0]
            sleep(0.001)
        except Exception as e:
            log(f"Error en GetFeed: {e}", "error")
            sleep(1)

def ClearRobotError(dashboard: DobotApiDashboard):
    """Hilo para manejo de errores del robot"""
    global robotErrorState
    if dashboard is None:
        return
        
    dataController, dataServo = alarmAlarmJsonFile()
    last_error = None
    error_count = 0
    
    while True:
        with globalLockValue:
            try:
                if robotErrorState:
                    numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
                    numbers = [int(num) for num in numbers]
                    
                    if numbers and numbers[0] == 0 and len(numbers) > 1:
                        current_error = ",".join(map(str, numbers[1:]))
                        
                        if current_error != last_error or error_count % 10 == 0:
                            for i in numbers[1:]:
                                if i == -2:
                                    log("! Colisión detectada", "error")
                                    continue
                                for item in dataController:
                                    if i == item["id"]:
                                        log(f"! Error controlador: {item['zh_CN']['description']}", "error")
                                        break
                                for item in dataServo:
                                    if i == item["id"]:
                                        log(f"! Error servo: {item['zh_CN']['description']}", "error")
                                        break
                            
                            last_error = current_error
                            error_count = 0
                            
                        dashboard.ClearError()
                        sleep(0.01)
                        dashboard.Continue()
                        error_count += 1
                else:
                    if enableStatus_robot is not None and algorithm_queue is not None:
                        if int(enableStatus_robot[0]) == 1 and int(algorithm_queue[0]) == 0:
                            dashboard.Continue()
            except Exception as e:
                if VERBOSE:
                    log(f"Error en ClearRobotError: {e}", "error")
        
        sleep(5)

def RunVision():
    """Hilo para el procesamiento de visión artificial"""
    global vision_active, model, cap
    vision_active = True
    
    while vision_active and cap is not None:
        ret, frame = cap.read()
        if not ret:
            log("× Error al leer cámara", "error")
            sleep(1)
            continue
            
        try:
            results = model.predict(source=frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
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
            cv2.imshow("Visión", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            log(f"× Error en visión: {str(e)}", "error")
            sleep(1)
            
    if cap:
        cap.release()
    cv2.destroyAllWindows()

def RunPoint(point_list: list):
    """Mueve el robot a una posición específica"""
    if not robot_connected or move is None:
        log("× Robot no conectado", "error")
        return False
        
    try:
        SpeedL(dashboard, velocidad)
        move.MovL(point_list[0], point_list[1], point_list[2], point_list[3])
        return True
    except Exception as e:
        log(f"× Error al mover robot: {str(e)}", "error")
        return False

def ActivarRobot():
    """Activa o desactiva el robot"""
    global Enable
    if not robot_connected or dashboard is None:
        log("× Robot no conectado", "error")
        return
        
    try:
        if Enable:
            result = dashboard.EnableRobot()
            if result[0] == "0":
                log("> Robot activado <")
                Enable = False
            else:
                log("× No se pudo activar robot", "error")
        else:
            result = dashboard.DisableRobot()
            if result[0] == "0":
                log("> Robot desactivado <")
                Enable = True
            else:
                log("× No se pudo desactivar robot", "error")
    except Exception as e:
        log(f"× Error al cambiar estado robot: {str(e)}", "error")

def MostrarMenu():
    """Muestra el menú de opciones mejorado"""
    print("\n" + "="*50)
    print("SISTEMA INTEGRADO ROBOT + VISIÓN")
    print("="*50)
    print(f"Estado Robot: {'Conectado' if robot_connected else 'Desconectado'}")
    print(f"Estado Visión: {'Disponible' if model else 'No disponible'}")
    print("="*50)
    print("1. Obtener posición actual")
    print("2. Activar/Desactivar robot")
    print("3. Mover a posición A")
    print("4. Mover a posición B")
    print("5. Limpiar errores")
    print("6. Cambiar velocidad")
    print("7. Iniciar/Detener visión")
    print("8. Reconectar robot")
    print("9. Salir")
    print("="*50)

def main():
    global vision_active, model, cap, velocidad, dashboard, move, feed, robot_connected
    
    # Inicializar sistemas
    ConnectRobot()
    InitializeVision()
    
    # Iniciar hilos del robot si está conectado
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.daemon = True
    feed_thread.start()
    
    error_thread = threading.Thread(target=ClearRobotError, args=(dashboard,))
    error_thread.daemon = True
    error_thread.start()
    
    if robot_connected:
        SpeedFactor(dashboard, 100)
    
    # Hilo de visión
    vision_thread = None
    
    while True:
        MostrarMenu()
        try:
            opcion = input("\nSeleccione una opción: ").strip()
            
            if opcion == "1" and robot_connected:
                try:
                    pose = dashboard.GetPose()
                    pos = pose.split(",")[1:pose.split(",").index('GetPose();')]
                    pos[0] = pos[0].replace("{", "")
                    pos[-1] = pos[-1].replace("}", "")
                    print(f"\nPosición actual: X:{pos[0]} Y:{pos[1]} Z:{pos[2]} R:{pos[3]}")
                except Exception as e:
                    log(f"× Error al obtener posición: {str(e)}", "error")
                    
            elif opcion == "2":
                ActivarRobot()
                
            elif opcion == "3":
                if RunPoint(posiciones["default"]):
                    print("> Movimiento a posición A iniciado <")
                    
            elif opcion == "4":
                if RunPoint(posiciones["B"]):
                    print("> Movimiento a posición B iniciado <")
                    
            elif opcion == "5" and robot_connected:
                try:
                    dashboard.ClearError()
                    print("> Errores limpiados <")
                except Exception as e:
                    log(f"× Error al limpiar errores: {str(e)}", "error")
                    
            elif opcion == "6" and robot_connected:
                try:
                    nueva_vel = int(input("Ingrese velocidad (1-100): "))
                    if 1 <= nueva_vel <= 100:
                        velocidad = nueva_vel
                        SpeedL(dashboard, velocidad)
                        print(f"> Velocidad cambiada a {velocidad}% <")
                    else:
                        print("× Velocidad fuera de rango", "error")
                except ValueError:
                    print("× Entrada inválida", "error")
                    
            elif opcion == "7" and model:
                if not vision_active:
                    vision_thread = threading.Thread(target=RunVision)
                    vision_thread.daemon = True
                    vision_thread.start()
                    print("> Visión iniciada <")
                else:
                    vision_active = False
                    if vision_thread:
                        vision_thread.join()
                    print("> Visión detenida <")
                    
            elif opcion == "8":
                print("\nReconectando con el robot...")
                if ConnectRobot():
                    if robot_connected:
                        SpeedFactor(dashboard, 100)
                    print("> Reconexión completada <")
                else:
                    print("× No se pudo reconectar", "error")
                    
            elif opcion == "9":
                print("\nFinalizando programa...")
                vision_active = False
                if vision_thread:
                    vision_thread.join()
                if robot_connected and not Enable:
                    dashboard.DisableRobot()
                break
                
            else:
                print("× Opción no válida o no disponible", "error")
                
        except Exception as e:
            log(f"× Error inesperado: {str(e)}", "error")
        
        input("\nPresione Enter para continuar...")

if __name__ == '__main__':
    main()