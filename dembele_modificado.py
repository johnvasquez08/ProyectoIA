import flet as ft
import threading
import os
from time import sleep
import numpy as np
import time
import re
import cv2
import base64
from io import BytesIO
from ultralytics import YOLO
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusIOException
from math import pi

# --- IMPORTA TU API DEL ROBOT ---
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile

# Variables de configuración MODBUS
slave_id = 100      # ID del esclavo
start_address = 8450  # Dirección de inicios
num_registers = 2    # Cantidad de registros a leer
radio_tambor = 58.705  # Radio del tambor en mm

# Configuración del cliente Modbus RTU
client = ModbusSerialClient(
    port='COM4',  # Cambiado al puerto COM4
    baudrate=9600,
    stopbits=1,
    bytesize=8,
    parity='N',
    timeout=1
)
# =========================
# VARIABLES GLOBALES
# =========================

current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()
Enable = True
Coord2Send = []
Pxu = -98.5
Pyu = 15.7
MatrizRUT = np.array([[0,1,0,211.841],[1,0,0,-254.42],[0,0,1,-152.26],[0,0,0,1]])
MatrizUPT = np.array([[0,-1,0,Pxu],[1,0,0,Pyu],[0,0,1,-65],[0,0,0,1]])

# Posiciones predefinidas
A = [360, 0, -42, 0]
PRUEBA = [360, 0, -127, 10]

# Visión artificial
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
expected_ids = [1, 2, 3, 4]
output_size = (400, 400)
calibrated = False
warp_matrix = None
detection_active = False
detection_thread = None
xanterior = 0
yanterior = 0
tiempo = 0
velocidadbanda = 0
angulo = 0
factordecorrecion = 0
contadorazul = 0
contadorrojo = 0
contadorblanco = 0
zheta = 0

# Estado de conexión del robot
robot_connected = False
dashboard = None
move = None
feed = None
feed_thread = None
error_thread = None

# Variable para el modo de operación: CLASIFICACION o CORRECCION
modo_operacion = "CLASIFICACION"

# Modelo YOLO
try:
    model = YOLO("C:/Users/johnd/Documents/proyectofinal/ProyectoIA/runsnuev/obb/train/weights/best.pt")
    vision_available = True
except Exception as e:
    print(f"Error al cargar el modelo YOLO: {e}")
    vision_available = False

detection_data = {
    "objeto": None,
    "centroide_x": None,
    "centroide_y": None,
    "angulo": None,
    "tiempo": None,
    "nueva_deteccion": False
}
detection_lock = threading.Lock()

# LOG
log_registros = []
def agregar_log(mensaje):
    log_registros.append(f"{time.strftime('%H:%M:%S')} - {mensaje}")

# =========================
# FUNCIONES DE ROBOT (igual que tu script, solo cambia los print/input por agregar_log y diálogos)
# =========================

# Aquí van todas tus funciones de robot, pero donde había print() pon agregar_log()
# Donde había input(), usa un diálogo (te muestro un ejemplo en el método ClearRobotError)

def ConnectRobot():
    global robot_connected, dashboard, move, feed
    try:
        ip = "192.168.0.12"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        agregar_log("Estableciendo conexión...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        agregar_log("Conexión exitosa")
        robot_connected = True
        return dashboard, move, feed
    except Exception as e:
        agregar_log(f"Conexión fallida - Error: {e}")
        robot_connected = False
        return None, None, None

def RunPoint(move, point_list):
    if not robot_connected or move is None:
        agregar_log("Error: El robot no está conectado")
        return False
    try:
        move.MovL(point_list[0], point_list[1], point_list[2], point_list[3])
        return True
    except Exception as e:
        agregar_log(f"Error al ejecutar el punto: {e}")
        return False
def RunArco(move: DobotApiMove, point_list: list):
    if not robot_connected or move is None:
        agregar_log("Error: El robot no está conectado")
        return False
    try:
        move.Arc(point_list[0], point_list[1], point_list[2], point_list[3],point_list[4], point_list[5], point_list[6], point_list[7])
    except Exception as e:
        agregar_log(f"Error al ejecutar el punto: {e}")
        return False

def GetFeed(feed):
    global current_actual, algorithm_queue, enableStatus_robot, robotErrorState, robot_connected
    hasRead = 0
    while robot_connected:
        try:
            data = bytes()
            while hasRead < 1440:
                temp = feed.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            hasRead = 0
            feedInfo = np.frombuffer(data, dtype=MyType)
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                globalLockValue.acquire()
                current_actual = feedInfo["tool_vector_actual"][0]
                algorithm_queue = feedInfo['isRunQueuedCmd'][0]
                enableStatus_robot = feedInfo['EnableStatus'][0]
                robotErrorState = feedInfo['ErrorStatus'][0]
                globalLockValue.release()
            sleep(0.001)
        except Exception as e:
            agregar_log(f"Error en la retroalimentación: {e}")
            robot_connected = False
            break
        sleep(0.001)

def WaitArrive(point_list):
    if not robot_connected:
        agregar_log("Error: El robot no está conectado")
        return False
    timeout = time.time() + 30
    while time.time() < timeout:
        is_arrive = True
        globalLockValue.acquire()
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
            if is_arrive:
                globalLockValue.release()
                return True
        globalLockValue.release()
        sleep(0.001)
    agregar_log("Tiempo de espera agotado")
    return False

def SpeedL(dashboard, speed):
    if not robot_connected or dashboard is None:
        agregar_log("Error: El robot no está conectado")
        return False
    try:
        string = "SpeedL({:d})".format(speed)
        return dashboard.sendRecvMsg(string)
    except Exception as e:
        agregar_log(f"Error al establecer la velocidad lineal: {e}")
        return False
def DO(dashboard, index, status):
        string = "DO({:d},{:d})".format(index, status)
        return dashboard.sendRecvMsg(string)

def ClearError(dashboard):
    if not robot_connected or dashboard is None:
        agregar_log("Error: El robot no está conectado")
        return False
    try:
        string = "ClearError()"
        return dashboard.sendRecvMsg(string)
    except Exception as e:
        agregar_log(f"Error al limpiar errores: {e}")
        return False

def ActivarRobot(dashboard, feed):
    global Enable, robot_connected, feed_thread
    if not robot_connected or dashboard is None:
        agregar_log("Error: El robot no está conectado")
        return False
    try:
        if Enable:
            ola = dashboard.EnableRobot()
            if ola[0] == "0":
                agregar_log("Robot habilitado con éxito")
                Enable = False
                if feed_thread is None or not feed_thread.is_alive():
                    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
                    feed_thread.daemon = True
                    feed_thread.start()
                return True
            else:
                agregar_log("No se pudo habilitar el robot")
                return False
        else:
            ola = dashboard.DisableRobot()
            if ola[0] == "0":
                agregar_log("Robot deshabilitado con éxito")
                Enable = True
                return True
            else:
                agregar_log("No se pudo deshabilitar el robot")
                return False
    except Exception as e:
        agregar_log(f"Error al activar/desactivar el robot: {e}")
        return False

def ObtenerPosicion(dashboard):
    if not robot_connected or dashboard is None:
        agregar_log("Error: El robot no está conectado")
        return None
    try:
        pose = dashboard.GetPose()
        posicionActual = pose.split(",")
        posicionActual = posicionActual[1:posicionActual.index('GetPose();')]
        posicionActual[0] = posicionActual[0].replace("{", "")
        posicionActual[-1] = posicionActual[-1].replace("}", "")
        agregar_log(f"Posición actual: {posicionActual}")
        return posicionActual
    except Exception as e:
        agregar_log(f"Error al obtener posición: {e}")
        return None
def SpeedFactor(dashboard: DobotApiDashboard, speed):
        
        string = "SpeedFactor({:d})".format(speed)
        return dashboard.sendRecvMsg(string)
def ReconectarRobot():
    global dashboard, move, feed, robot_connected, feed_thread, error_thread
    robot_connected = False
    if feed_thread and feed_thread.is_alive():
        feed_thread.join(timeout=1.0)
    if error_thread and error_thread.is_alive():
        error_thread.join(timeout=1.0)
    agregar_log("Intentando reconectar al robot...")
    dashboard, move, feed = ConnectRobot()
    if robot_connected:
        feed_thread = threading.Thread(target=GetFeed, args=(feed,))
        feed_thread.daemon = True
        feed_thread.start()
        agregar_log("Reconexión exitosa")
        return True
    else:
        agregar_log("Reconexión fallida")
        return False

# =========================
# FUNCIÓN PARA CAMBIAR EL MODO DE OPERACIÓN
# =========================

def cambiar_modo_operacion():
    global modo_operacion
    if modo_operacion == "CLASIFICACION":
        modo_operacion = "CORRECCION"
        agregar_log(f"Modo cambiado a: {modo_operacion}")
    else:
        modo_operacion = "CLASIFICACION"
        agregar_log(f"Modo cambiado a: {modo_operacion}")
    return modo_operacion

# =========================
# FUNCIONES DE VISIÓN ARTIFICIAL
# =========================

def setup_workspace():
    global calibrated, warp_matrix
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    agregar_log("Coloque los marcadores ArUco (1-4) en las esquinas del área de trabajo.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            ids = ids.flatten()
            if all(id_ in ids for id_ in expected_ids):
                detected = {id_: corner[0][0] for id_, corner in zip(ids, corners)}
                points = np.array([detected[id_] for id_ in expected_ids], dtype=np.float32)
                src_pts = np.array([detected[id_] for id_ in expected_ids], dtype=np.float32)
                dst_pts = np.array([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]], dtype=np.float32)
                warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                calibrated = True
                agregar_log("Área de trabajo calibrada correctamente.")
                break
    cap.release()

# Para mostrar frames en Flet (convertir frame OpenCV a base64 PNG)
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.png', frame)
    return base64.b64encode(buffer).decode('utf-8')

def get_vel():
    global vel
    result = client.read_holding_registers(address=start_address, count=num_registers, slave=slave_id)
    if result.isError():
        print(f"Error al leer los registros: {result}")
    else:
        # Imprimir solo el segundo valor del registro
        segundo_registro = result.registers[1]  # Índice 1 para el segundo registro
        RPM = segundo_registro / 100
        Rads = RPM * ((2 * pi) / 60)
        vel = Rads * radio_tambor  # Velocidad en mm/s
        vel = round(vel,3)
        agregar_log(f"Velocidad de la banda: {vel}")
        return vel

# Detección en hilo
def detection_loop(image_update_callback):
    global warp_matrix,modo_operacion, calibrated, detection_active, detection_data, Coord2Send,contadorazul,zheta, xanterior, tiempo, velocidadbanda, angulo, factordecorrecion, contadorblanco,contadorrojo, vel, modo_operacion
    if not calibrated:
        agregar_log("[!] El área de trabajo no está calibrada. Configure primero.")
        return
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    detection_active = True
    while detection_active:
        ret, frame = cap.read()
        if not ret:
            break
        warped = cv2.warpPerspective(frame, warp_matrix, output_size)
        results = model.predict(source=warped, conf=0.5)
        annotated = results[0].plot()
        # Procesar detecciones
        if results[0].obb is not None and results[0].obb.data is not None:
            detections = results[0].obb.data
            for det in detections:
                cx, cy, w, h, theta, conf, class_id = det[:7].tolist()
                class_id = int(class_id)
                class_name = model.names[class_id]
                if 147 <= cy <= 250:
                    if len(Coord2Send) != 2:
                        with detection_lock:
                            detection_data["objeto"] = class_name
                            detection_data["centroide_x"] = int(cx)
                            Coord2Send.append(int(cx))
                            detection_data["centroide_y"] = int(cy)
                            Coord2Send.append(int(cy))
                            detection_data["angulo"] = round(theta, 2)
                            detection_data["tiempo"] = time.time()
                            detection_data["nueva_deteccion"] = True
                    if len(Coord2Send) == 2:
                        CentimetroYCamara = (((Coord2Send[1] * 20.6)/400)*10)
                        angulo = int(float(180/pi)* float(theta))
                        CentrimetroY = (-350)
                        CentrimetroX = (((Coord2Send[0] * 26.2)/400)*10)
                        velocidadbanda = 98.9
                        if velocidadbanda >= 60 and velocidadbanda <= 90 and cx <= 121:
                            factordecorrecion = 25
                        elif velocidadbanda >= 60 and velocidadbanda <= 90 and cx > 121:
                            factordecorrecion = 23
                        if velocidadbanda >= 91 and velocidadbanda <= 130 and cx <= 121:
                            factordecorrecion = 17
                        elif velocidadbanda >= 91 and velocidadbanda <= 130 and cx > 121:
                            factordecorrecion = 13
                        if velocidadbanda >= 131 and velocidadbanda <= 175 and cx <= 121:
                            factordecorrecion = 16
                        elif velocidadbanda >= 131 and velocidadbanda <= 175 and cx > 121 and cx >= 220:
                            factordecorrecion = 14
                        elif velocidadbanda >= 131 and velocidadbanda <= 175 and cx > 220:
                            factordecorrecion = 11
                        if velocidadbanda >= 176 and velocidadbanda <= 205 and cx <= 121:
                            factordecorrecion = 6
                        elif velocidadbanda >= 176 and velocidadbanda <= 205 and cx > 121:
                            factordecorrecion = 2
                        
                        tiempo = (350+CentimetroYCamara)/velocidadbanda
                        tiempodelay = tiempo - 0.8
                        agregar_log(f"Centimetro Y Camara es: {CentimetroYCamara} ")
                        agregar_log(f"El tiempo de delay fue: {tiempodelay} segundos")
                        P = np.array([[CentrimetroX],[CentrimetroY],[28], [1]])
                        paso1 = np.dot(MatrizRUT,MatrizUPT)
                        Pos = np.dot(paso1, P)
                        valores = Pos.flatten().tolist()
                        xactual, yactual, z, w = Pos.flatten()
                        CoordsPrevias = [xactual, yactual+factordecorrecion, -42, 0]
                        CoordsPreviasCorregidas = [xactual, yactual+factordecorrecion, -42, angulo]
                        """CoordsDejada = [xactual, yactual+factordecorrecion, -120, angulo]"""
                        CoordsDejada = [360, 0, -120, angulo]
                        Coords = [xactual, yactual+factordecorrecion, -129, 0]
                        A2 = [360, 0, -42, angulo]
                        B = [223,205,53,angulo,40,283,15,angulo]
                        if class_name == "Azul":
                            descuento = -10*contadorazul
                            zheta = -140-descuento
                            contadorazul += 1
                            C = [46,289,zheta,angulo]
                            agregar_log(f"El zheta es: {zheta}")
                        elif class_name == "Rojo":
                            descuento = -10*contadorrojo
                            zheta = -140-descuento
                            contadorrojo += 1
                            C = [-56,285,zheta,angulo]
                        elif class_name == "Blanco":
                            descuento = -10*contadorblanco
                            zheta = -140-descuento
                            contadorblanco += 1
                            C = [-11,394,zheta,angulo]
                        D = [40,283,15,angulo]
                        DA = [223,205,53,angulo,360, 0, -42, 0]
                        agregar_log(f"Class ID {class_name}")
                         # Usar el modo_operacion como condicional para el procesamiento
                        if modo_operacion == "CLASIFICACION":    
                            if abs(abs(xactual)-abs(xanterior)) > 3:
                                agregar_log(f"El modo de operacion es {modo_operacion}")
                                SpeedFactor(dashboard,100)
                                SpeedL(dashboard,90)
                                RunPoint(move, CoordsPrevias)
                                DO(dashboard,1,1)
                                time.sleep(tiempodelay)
                                t1=time.time()
                                RunPoint(move, Coords)
                                t2= time.time()
                                tfinal = t2 - t1
                                agregar_log(f"El tiempo fue {tfinal: .4f}")
                                """RunPoint(move, CoordsPreviasCorregidas)"""
                                RunPoint(move, CoordsPreviasCorregidas)
                                RunPoint(move, A2)
                                RunArco(move, B)
                                RunPoint(move, C)
                                WaitArrive(C)
                                DO(dashboard,1,0)
                                time.sleep(0.3)
                                DO(dashboard,2,1)
                                time.sleep(1)
                                DO(dashboard,2,0)
                                RunPoint(move, D)
                                RunArco(move, DA)
                        elif modo_operacion == "CORRECCION":
                            if abs(abs(xactual)-abs(xanterior)) > 3:
                                agregar_log(f"El modo de operacion es {modo_operacion}")
                                SpeedFactor(dashboard,100)
                                SpeedL(dashboard,90)
                                RunPoint(move, CoordsPrevias)
                                DO(dashboard,1,1)
                                time.sleep(tiempodelay)
                                t1=time.time()
                                RunPoint(move, Coords)
                                t2= time.time()
                                tfinal = t2 - t1
                                agregar_log(f"El tiempo fue {tfinal: .4f}")
                                """RunPoint(move, CoordsPreviasCorregidas)"""
                                RunPoint(move, CoordsPreviasCorregidas)
                                RunPoint(move, A2)
                                RunPoint(move, CoordsDejada)
                                WaitArrive(CoordsDejada)
                                DO(dashboard,1,0)
                                time.sleep(0.8)
                                DO(dashboard,2,1)
                                time.sleep(2)
                                DO(dashboard,2,0)
                                RunPoint(move, A2)
                                
                        Coord2Send=[]
                        xanterior = xactual
                        yanterior = yactual
                
                cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                cv2.putText(annotated, f"Angulo({angulo}),({int(cx)}, {int(cy)}),{class_name}", (int(cx)+10, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # Actualizar imagen en la UI
        if image_update_callback:
            image_update_callback(frame_to_base64(annotated))
        sleep(0.03)
    detection_active = False
    cap.release()

# =========================
# FLET UI
# =========================

def main(page: ft.Page):
    page.title = "Control Robot Dobot - Flet"
    page.window_width = 1000
    page.window_height = 700
    loading_overlay = ft.Container(
        ft.ProgressRing(),
        alignment=ft.alignment.center,
        bgcolor=ft.colors.with_opacity(0.3, ft.colors.BLACK),
        expand=True
    )

    # LOG
    log_view = ft.ListView(expand=True, spacing=2, padding=10, auto_scroll=True)
    def actualizar_log():
        log_view.controls.clear()
        for l in log_registros[-100:]:
            log_view.controls.append(ft.Text(l, size=12))
        page.update()

    # ===========
    # CONTROL DE ROBOT
    # ===========
    velocidad = ft.TextField(label="Velocidad", value="50", width=100)
    # Texto que muestra el modo actual
    modo_texto = ft.Text(f"Modo actual: {modo_operacion}", size=16, weight=ft.FontWeight.BOLD)
    
    # Botón para cambiar el modo
    def on_cambiar_modo(e):
        global modo_operacion
        modo_operacion = cambiar_modo_operacion()
        modo_texto.value = f"Modo actual: {modo_operacion}"
        # Cambiar color del botón según el modo
        if modo_operacion == "CLASIFICACION":
            cambiar_modo_btn.bgcolor = ft.colors.GREEN
            cambiar_modo_btn.text = "Cambiar a CORRECCIÓN"
        else:
            cambiar_modo_btn.bgcolor = ft.colors.AMBER
            cambiar_modo_btn.text = "Cambiar a CLASIFICACIÓN"
        page.update()
        actualizar_log()
    
    # Botón inicial con color según el modo predeterminado
    cambiar_modo_btn = ft.ElevatedButton(
        "Cambiar a CORRECCIÓN", 
        on_click=on_cambiar_modo,
        bgcolor=ft.colors.GREEN
    )
    
    def on_succion_on(e):
        DO(dashboard, 1, 1)
        agregar_log("Succión activada")
        actualizar_log()

    def on_succion_off(e):
        DO(dashboard, 1, 0)
        agregar_log("Succión desactivada")
        actualizar_log()

    def on_expulsion_on(e):
        DO(dashboard, 2, 1)
        agregar_log("Expulsión activada")
        actualizar_log()

    def on_expulsion_off(e):
        DO(dashboard, 2, 0)
        agregar_log("Expulsión desactivada")
        actualizar_log()

    def on_obtener_posicion(e):
        ObtenerPosicion(dashboard)
        actualizar_log()
    def on_activar_robot(e):
        ActivarRobot(dashboard, feed)
        actualizar_log()
    def on_posicion_a(e):
        SpeedFactor(dashboard,100)
        SpeedL(dashboard, 90)
        if RunPoint(move, A):
            WaitArrive(A)
            agregar_log("Movido a posición A")
        else:
            agregar_log("No se pudo ejecutar el movimiento a A")
        actualizar_log()
    def on_prueba(e):
        SpeedFactor(dashboard,100)
        SpeedL(dashboard, 90)
        if RunPoint(move, PRUEBA):
            WaitArrive(PRUEBA)
            agregar_log("Movido a PRUEBA")
        else:
            agregar_log("No se pudo ejecutar el movimiento PRUEBA")
        actualizar_log()
    def on_limpiar_error(e):
        ClearError(dashboard)
        actualizar_log()
    def on_reconectar(e):
        ReconectarRobot()
        actualizar_log()
    control_robot = ft.Column([
        # Añadir el texto del modo y el botón para cambiar el modo
        ft.Row([
            modo_texto,
            cambiar_modo_btn
        ]),
        ft.Row([
            ft.ElevatedButton("Obtener posición", on_click=on_obtener_posicion),
            ft.ElevatedButton("Activar/Desactivar", on_click=on_activar_robot),
            ft.ElevatedButton("Posición A", on_click=on_posicion_a),
            ft.ElevatedButton("Posición PRUEBA", on_click=on_prueba),
            ft.ElevatedButton("Limpiar Error", on_click=on_limpiar_error),
            
        ]),
        ft.Row([
            ft.Text("Velocidad:"),
            velocidad,
        ]),
        ft.Row([
            ft.ElevatedButton("Reconectar Robot", on_click=on_reconectar)
        ]),
        ft.Row([
        ft.ElevatedButton("Succión ON", on_click=on_succion_on, bgcolor=ft.colors.GREEN_200),
        ft.ElevatedButton("Succión OFF", on_click=on_succion_off, bgcolor=ft.colors.RED_200),
        ft.ElevatedButton("Expulsión ON", on_click=on_expulsion_on, bgcolor=ft.colors.BLUE_200),
        ft.ElevatedButton("Expulsión OFF", on_click=on_expulsion_off, bgcolor=ft.colors.ORANGE_200),
    ]),
    ], spacing=10)

    # ===========
    # VISIÓN ARTIFICIAL
    # ===========

    # Imagen de cámara
    img_view = ft.Image(width=400, height=400, fit=ft.ImageFit.CONTAIN)
    def update_img(base64str):
        img_view.src_base64 = base64str
        page.update()

    def on_configurar_area(e):
        agregar_log("Configurando área de trabajo...")
        # Mostrar overlay de carga
        page.overlay.append(loading_overlay)
        page.update()
        setup_workspace()
        # Quitar overlay de carga
        if loading_overlay in page.overlay:
            page.overlay.remove(loading_overlay)
            page.update()
        actualizar_log()
    def on_iniciar_deteccion(e):
        global detection_thread, detection_active
        if detection_active:
            agregar_log("La detección ya está activa.")
            actualizar_log()
            return
        detection_thread = threading.Thread(target=detection_loop, args=(update_img,))
        detection_thread.daemon = True
        detection_thread.start()
        agregar_log("Detección iniciada")
        actualizar_log()
    def on_detener_deteccion(e):
        global detection_active
        detection_active = False
        agregar_log("Detección detenida")
        actualizar_log()
    vision_artificial = ft.Column([
        ft.Row([
            ft.ElevatedButton("Configurar área de trabajo", on_click=on_configurar_area),
            ft.ElevatedButton("Iniciar detección", on_click=on_iniciar_deteccion),
            ft.ElevatedButton("Detener detección", on_click=on_detener_deteccion),
        ]),
        img_view,
    ], spacing=10)

    # ===========
    # LOG DE REGISTROS
    # ===========
    log_panel = ft.Container(log_view, height=500, border=ft.border.all(1, "grey"))

    # ===========
    # TABS
    # ===========
    tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="Control de Robot", content=control_robot),
            ft.Tab(text="Visión Artificial", content=vision_artificial),
            ft.Tab(text="Log de Registros", content=log_panel),
        ],
        expand=1
    )
    page.add(tabs)

    # Actualización periódica del log
    def update_loop():
        while True:
            actualizar_log()
            sleep(1)
    threading.Thread(target=update_loop, daemon=True).start()

    # Conexión inicial
    def conectar_inicial():
        global dashboard, move, feed, robot_connected, feed_thread, error_thread
        dashboard, move, feed = ConnectRobot()
        if robot_connected:
            feed_thread = threading.Thread(target=GetFeed, args=(feed,))
            feed_thread.daemon = True
            feed_thread.start()
            agregar_log("Robot conectado y retroalimentación iniciada.")
        else:
            agregar_log("No se pudo conectar el robot al inicio.")
    threading.Thread(target=conectar_inicial, daemon=True).start()

ft.app(target=main)