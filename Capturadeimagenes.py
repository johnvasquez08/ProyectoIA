import cv2
import os
from datetime import datetime

# Configuración inicial
output_folder = "C:/Users/johnd/Documents/proyectofinal/ProyectoIA/Fotos"  # Carpeta donde se guardarán las fotos
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicializar la cámara
cap = cv2.VideoCapture(3)

# Configurar resolución (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Presiona 's' para capturar una foto")
print("Presiona 'q' para salir")

photo_count = 0

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo recibir el frame. Saliendo...")
        break
    
    # Mostrar el frame
    cv2.imshow('Camara - Presiona "s" para capturar', frame)
    
    # Esperar por teclas
    key = cv2.waitKey(1)
    
    if key == ord('q'):  # Salir
        break
    elif key == ord('s'):  # Capturar foto
        # Generar nombre de archivo con fecha y hora
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_folder, f"foto_{timestamp}_{photo_count}.jpg")
        
        # Guardar la imagen
        cv2.imwrite(filename, frame)
        print(f"Foto guardada como: {filename}")
        photo_count += 1
        
        # Mostrar mensaje en pantalla (opcional)
        cv2.putText(frame, "Foto guardada!", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camara - Presiona "s" para capturar', frame)
        cv2.waitKey(500)  # Mostrar mensaje por 500ms

# Liberar recursos
cap.release()
cv2.destroyAllWindows()