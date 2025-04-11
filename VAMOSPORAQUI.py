import cv2
import numpy as np
from ultralytics import YOLO
import time

# Configuración de ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
expected_ids = [1, 2, 3, 4]  # Marcadores esperados
output_size = (400, 400)  # Tamaño del área transformada


# Modelo YOLO
model = YOLO("runs/obb/train/weights/best.pt")

# Estado
calibrated = False
warp_matrix = None


def show_menu():
    print("\n--- MENÚ PRINCIPAL ---")
    print("1. Configurar área de trabajo")
    print("2. Iniciar detección en el área de trabajo")
    print("3. Salir")
    return input("Seleccione una opción: ")


def setup_workspace():
    global calibrated, warp_matrix
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print("\nColoque los marcadores ArUco (1-4) en las esquinas del área de trabajo")
    print("Presione 's' para seleccionar el área cuando todos sean visibles, o 'q' para cancelar")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if all(id_ in ids for id_ in expected_ids):
                detected = {id_: corner[0][0] for id_, corner in zip(ids, corners)}
                points = np.array([detected[id_] for id_ in expected_ids], dtype=np.float32)
                for i in range(4):
                    cv2.line(frame, tuple(points[i].astype(int)),
                             tuple(points[(i+1)%4].astype(int)), (0, 255, 255), 2)

        cv2.imshow("Configuracion del Area", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and ids is not None and all(id_ in ids for id_ in expected_ids):
            detected = {id_: corner[0][0] for id_, corner in zip(ids, corners)}
            src_pts = np.array([detected[id_] for id_ in expected_ids], dtype=np.float32)
            dst_pts = np.array([[0, 0], [output_size[0], 0],
                                [output_size[0], output_size[1]], [0, output_size[1]]], dtype=np.float32)
            warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            calibrated = True
            cv2.destroyWindow("Configuracion del Area")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detection_loop():
    global warp_matrix, calibrated

    if not calibrated:
        print("\n[!] El área de trabajo no está calibrada. Configure primero.")
        return

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print("\nDetección activa. Presione 'q' para volver al menú.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar transformación de perspectiva
        warped = cv2.warpPerspective(frame, warp_matrix, output_size)

        # Detección con YOLO sobre el área transformada
        results = model.predict(source=warped, conf=0.5)
        annotated = results[0].plot()

        # Dibujar centroides
        if results[0].obb is not None and results[0].obb.data is not None:
            detections = results[0].obb.data
            for det in detections:
    # Los datos OBB suelen ser: [x, y, ancho, alto, ángulo, confianza, class_id]
                cx, cy, w, h, theta, conf, class_id = det[:7].tolist()  # Ajustar índices
                class_id = int(class_id)  # Asegurar que es entero
                class_name = model.names[class_id]  # Obtener nombre correcto
                
                if 147 <= cy <= 270:
                    centroideguardadoX = cx
                    objetodetectado = class_name  # Clase actualizada
                    theta = theta
                    centroideguardadoY = cy
                    tiempo = time.time()
                if 'objetodetectado' in locals():
                    info_text = [
                        f"Objeto: {objetodetectado}",
                        f"Centroide X: {int(centroideguardadoX)}",
                        f"Centroide Y: {int(centroideguardadoY)}",
                        f"Ángulo: {round(theta, 2)}"
                    ]
                    for i, text in enumerate(info_text):
                        cv2.putText(annotated, text, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 255), 2)

                cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                cv2.putText(annotated, f"{class_name} ({int(cx)}, {int(cy)})", (int(cx)+10, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Detección en Área de Trabajo", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Bucle principal
while True:
    option = show_menu()
    if option == '1':
        setup_workspace()
    elif option == '2':
        detection_loop()
    elif option == '3':
        print("Saliendo...")
        break
    else:
        print("Opción no válida.")
