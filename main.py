import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile
from time import sleep
import numpy as np
import time
import re
import os

# Variables globales
current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()
Enable = True  # Variable para controlar el estado de activación del robot
A = [315, -11, 70, 10]
REC = [400, 0, -121, 0]
BC = [270, -154, 77, -32,4, -313, 111, -80]
AC = [270, -154, 77, -32,315, -11, 70, 10]
CABAJO = [2, -295, 39, -78]
C = [4, -313, 111, -80]
PRUEBA = [395.3,-111.12,-120,0]

def ConnectRobot():
    try:
        ip = "192.168.0.12"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        print("Estableciendo conexión...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        print(">.<Conexión exitosa>!<")
        return dashboard, move, feed
    except Exception as e:
        print(":(Conexión fallida:(")
        raise e

def RunPoint(move: DobotApiMove, point_list: list):
    move.MovL(point_list[0], point_list[1], point_list[2], point_list[3])
def RunArco(move: DobotApiMove, point_list: list):
    move.Arc(point_list[0], point_list[1], point_list[2], point_list[3],point_list[4], point_list[5], point_list[6], point_list[7])

def GetFeed(feed: DobotApi):
    global current_actual
    global algorithm_queue
    global enableStatus_robot
    global robotErrorState
    hasRead = 0
    while True:
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
                # Refresh Properties
                current_actual = feedInfo["tool_vector_actual"][0]
                algorithm_queue = feedInfo['isRunQueuedCmd'][0]
                enableStatus_robot = feedInfo['EnableStatus'][0]
                robotErrorState = feedInfo['ErrorStatus'][0]
                globalLockValue.release()
            sleep(0.001)
        except Exception as e:
            print(f"Error en GetFeed: {e}")
            sleep(1)  # Esperar antes de intentar de nuevo

def WaitArrive(point_list):
    while True:
        is_arrive = True
        globalLockValue.acquire()
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
            if is_arrive:
                globalLockValue.release()
                return
        globalLockValue.release()  
        sleep(0.001)
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
def DO(dashboard, index, status):
        """
    Set digital signal output (Queue instruction)
    index : Digital output index (Value range:1~24)
    status : Status of digital signal output port(0:Low level，1:High level
    """
        string = "DO({:d},{:d})".format(index, status)
        return dashboard.sendRecvMsg(string)

def ClearRobotError(dashboard: DobotApiDashboard):
    global robotErrorState, enableStatus_robot, algorithm_queue
    dataController, dataServo = alarmAlarmJsonFile()    # Leer códigos de alarma
    while True:
        globalLockValue.acquire()
        try:
            if robotErrorState:
                numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
                numbers = [int(num) for num in numbers]
                if (numbers[0] == 0):
                    if (len(numbers) > 1):
                        for i in numbers[1:]:
                            alarmState = False
                            if i == -2:
                                print("Alarma: Colisión detectada ", i)
                                alarmState = True
                            if alarmState:
                                continue                
                            for item in dataController:
                                if i == item["id"]:
                                    print("Alarma de controlador:", i, item["zh_CN"]["description"])
                                    alarmState = True
                                    break 
                            if alarmState:
                                continue
                            for item in dataServo:
                                if i == item["id"]:
                                    print("Alarma de servo:", i, item["zh_CN"]["description"])
                                    break  
                           
                        choose = input("Presione 1 para borrar el error y continuar: ")     
                        if int(choose) == 1:
                            dashboard.ClearError()
                            sleep(0.01)
                            dashboard.Continue()
            else:
                if enableStatus_robot is not None and algorithm_queue is not None:
                    if int(enableStatus_robot[0]) == 1 and int(algorithm_queue[0]) == 0:
                        dashboard.Continue()
        except Exception as e:
            print(f"Error en ClearRobotError: {e}")
        finally:
            globalLockValue.release()
            sleep(5)

def ActivarRobot(dashboard: DobotApiDashboard, feed):
    global Enable
    if Enable:
        ola = dashboard.EnableRobot()
        feed_thread = threading.Thread(target=GetFeed, args=(feed,))
        feed_thread.daemon = True  # Usar .daemon en lugar de setDaemon()
        feed_thread.start()

        if ola[0] == "0":
            print("Robot habilitado con éxito")
            Enable = False
        else:
            print("No se pudo habilitar el robot")
    else:
        ola = dashboard.DisableRobot()
       
        if ola[0] == "0":
            print("Robot deshabilitado con éxito")
            Enable = True
        else:
            print("No se pudo deshabilitar el robot")

def ObtenerPosicion(dashboard):
    try:
        pose = dashboard.GetPose()
        posicionActual = pose.split(",")
        posicionActual = posicionActual[1:posicionActual.index('GetPose();')]
        posicionActual[0] = posicionActual[0].replace("{", "")
        posicionActual[-1] = posicionActual[-1].replace("}", "")
        print("\nPosición actual:", posicionActual)
    except Exception as e:
        print(f"Error al obtener posición: {e}")

def MostrarMenu():
    print("\n" + "="*50)
    print("MENÚ DE CONTROL DEL ROBOT DOBOT")
    print("="*50)
    print("1. Obtener posición actual")
    print("2. Activar/Desactivar robot")
    print("3. Posicion A")
    print("4. Limpiar Error")
    print("5. Salir")
    print("6. Posicion B")
    print("7. Cambiar velocidad")
    print("="*50)

def LimpiarPantalla():
    # Limpiar pantalla según el sistema operativo
    os.system('cls' if os.name == 'nt' else 'clear')

def ClearError(dashboard: DobotApiDashboard):
        """
    Clear controller alarm information
    """
        string = "ClearError()"
        return dashboard.sendRecvMsg(string)

if __name__ == '__main__':
    try:
        dashboard, move, feed = ConnectRobot()
        
        # Iniciar hilo de retroalimentación con daemon=True
        feed_thread = threading.Thread(target=GetFeed, args=(feed,))
        feed_thread.daemon = True  # Usar .daemon en lugar de setDaemon()
        feed_thread.start()
        
        # Iniciar hilo para manejar errores
        error_thread = threading.Thread(target=ClearRobotError, args=(dashboard,))
        error_thread.daemon = True  # Usar .daemon en lugar de setDaemon()
        error_thread.start()
        posiciondefault = [248, 40, 0, 0]
        point_a = [284, -66, -7, 0]
        point_b = [293, -66, -142, -13]
        point_c = [284, -66, -7, 0]
        point_d = [277, 140, 6, 30]
        point_e = [277, 140, -120, 30]
        SpeedFactor(dashboard, 100)
        velocidad = 50
        
        
        while True:
            MostrarMenu()
            opcion = input("\nSeleccione una opción (1-5): ")
            
            if opcion == "1":
                ObtenerPosicion(dashboard)
                input("\nPresione Enter para continuar...")
                
            elif opcion == "2":
                ActivarRobot(dashboard, feed)
                input("\nPresione Enter para continuar...")
                
            elif opcion == "3":
                    SpeedL(dashboard, velocidad)
                    tiempo_inicio = time.time()
                    RunPoint(move, point_a)
                    WaitArrive(point_a)
                    tiempo_fin = time.time()
                    tiempofinal = tiempo_fin - tiempo_inicio
                    print(f"Tiempo transcurrido: {tiempofinal: .4f} segundos")
            elif opcion == "6":
                    SpeedL(dashboard, 80)
                    empezo = time.time()
                    RunPoint(move, PRUEBA)
                    WaitArrive(PRUEBA)
                    termino = time.time()
                    tiempofinal = termino - empezo
                    RunPoint(move, A)
                    print(f"Tiempo transcurrido: {tiempofinal: .4f} segundos")                    
       

                    
            elif opcion == "4":
                ClearError(dashboard)
                    
                
            elif opcion == "5":
                print("\nFinalizando programa...")
                # Asegurarse de deshabilitar el robot antes de salir
                if not Enable:
                    dashboard.DisableRobot()
                break
            elif opcion == "7":
                velocidad = int(input("Nueva velocidad: "))
                
            else:
                print("\nOpción no válida. Intente de nuevo.")
                input("\nPresione Enter para continuar...")
                
    except Exception as e:
        print(f"Error en el programa principal: {e}")
    finally:
        print("Programa finalizado")
