import numpy as np
Pxu = -98.5
Pyu = 15.7
MatrizRUT = np.array([[0,1,0,211.841],
                     [1,0,0,-254.42],
                     [0,0,1,-152.26],
                     [0,0,0,1]])

MatrizUPT = np.array([[0,-1,0,Pxu],
                     [1,0,0,Pyu],
                     [0,0,1,-65],
                     [0,0,0,1]])

P = np.array([[167.76],[-241.8],[28], [1]])

paso1 = np.dot(MatrizRUT,MatrizUPT)

Pos = np.dot(paso1, P)

print(Pos)