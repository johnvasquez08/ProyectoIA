import numpy as np
from tensorflow.keras.models import load_model
import joblib

def main():
    # Cargar modelo y scaler
    model = load_model('robot_picking_model.h5')
    scaler = joblib.load('scaler.save')
    
    print("=== PRUEBA INTERACTIVA DEL MODELO ===")
    print("Ingresa los valores solicitados (deja vacío para salir)\n")
    
    while True:
        try:
            # Solicitar datos al usuario
            print("\n--- Nuevo Cálculo ---")
            centroid_x = input("Posición X del centroide: ")
            if not centroid_x: break
            
            centroid_y = input("Posición Y del centroide: ")
            if not centroid_y: break
            
            speed = input("Velocidad de la banda: ")
            if not speed: break
            
            robot_x = input("Posición X del robot: ")
            if not robot_x: break
            
            robot_y = input("Posición Y del robot: ")
            if not robot_y: break
            
            robot_z = input("Posición Z del robot: ")
            if not robot_z: break
            
            # Convertir a números
            centroid = [float(centroid_x), float(centroid_y)]
            belt_speed = float(speed)
            robot_pos = [float(robot_x), float(robot_y), float(robot_z)]
            
            # Preparar entrada
            input_data = np.array([
                centroid[0], centroid[1], 
                belt_speed,
                robot_pos[0], robot_pos[1], robot_pos[2]
            ]).reshape(1, -1)
            
            # Normalizar y predecir
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            
            # Mostrar resultados
            print("\nResultado de la predicción:")
            print(f"Posición X para recoger: {prediction[0][0]:.2f}")
            print(f"Posición Y para recoger: {prediction[0][1]:.2f}")
            print(f"Posición Z para recoger: {prediction[0][2]:.2f}")
            
        except ValueError:
            print("Error: Ingresa valores numéricos válidos. Intenta nuevamente.")
        except Exception as e:
            print(f"Error inesperado: {str(e)}")
    
    print("\nPrueba finalizada.")

if __name__ == "__main__":
    main()