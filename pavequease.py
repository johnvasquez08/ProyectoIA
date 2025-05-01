import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Definir las variables difusas
velocidadbanda = ctrl.Antecedent(np.arange(60, 220, 1), 'VelocidadBanda')  # Rango de 60 a 205
factordecorrecion = ctrl.Consequent(np.arange(2, 17, 1), 'FactorDeCorrecion')  # Rango de 3 a 15

# 2. Generar las funciones de membresía para la entrada
velocidadbanda['baja'] = fuzz.trimf(velocidadbanda.universe, [60, 61, 86])
velocidadbanda['media'] = fuzz.trimf(velocidadbanda.universe, [86, 100, 125])
velocidadbanda['alta'] = fuzz.trimf(velocidadbanda.universe, [125, 150, 175])
velocidadbanda['muy_alta'] = fuzz.trimf(velocidadbanda.universe, [175, 200, 220])

# 3. Funciones de membresía para la salida
factordecorrecion['muy_alto'] = fuzz.trimf(factordecorrecion.universe, [12, 15, 17])
factordecorrecion['alto'] = fuzz.trimf(factordecorrecion.universe, [7, 8, 12])
factordecorrecion['medio'] = fuzz.trimf(factordecorrecion.universe, [4, 5.5, 7])
factordecorrecion['bajo'] = fuzz.trimf(factordecorrecion.universe, [2, 3, 4])


# 4. Visualizar las funciones de membresía
velocidadbanda.view()
factordecorrecion.view()

# 5. Definir las reglas difusas (similares a tus ifs)
regla1 = ctrl.Rule(velocidadbanda['baja'], factordecorrecion['muy_alto'])
regla2 = ctrl.Rule(velocidadbanda['media'], factordecorrecion['alto'])
regla3 = ctrl.Rule(velocidadbanda['alta'], factordecorrecion['medio'])
regla4 = ctrl.Rule(velocidadbanda['muy_alta'], factordecorrecion['bajo'])

# 6. Crear el sistema de control difuso
sistema_control = ctrl.ControlSystem([regla1, regla2, regla3, regla4])
simulador = ctrl.ControlSystemSimulation(sistema_control)

# 7. Probar con un valor de velocidadbanda, por ejemplo 120
simulador.input['VelocidadBanda'] = 200
simulador.compute()
print(f"Factor de correción: {simulador.output['FactorDeCorrecion']}")

# 8. Visualizar el resultado
factordecorrecion.view(sim=simulador)
plt.show()
