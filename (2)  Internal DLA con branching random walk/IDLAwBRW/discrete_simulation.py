import BRW

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

""" DEPRECATED, archivo principal (LIBRERIA): BRW
Herramienta de animacion: animar.py"""



# SETUP
r_inicial = 3000# 
""" Se usa para establecer una cantidad inicial de partículas antes de la animación
CONTROLARSE: ESTO NO REPRESENTA PARA NADA NINGUN TIPO DE RADIO DEL CLUSTER
ES LA CANTIDAD DE PARTICULAS CON LAS QUE SE INICIA LA SIMULACION (SE SUELTAN DE A UNA)
R GRANDE -> SE PUEDE ALCANZAR ESTADO SUPER-CRITICO Y LA ANIMACIÓN NUNCA PARTE"""

p = 0.5

""" Probabilidad de aparearse (Implementado como una una bernoulli en cada dt de la simulacion)
igual es como mitosis pero waterer"""


# Crear figura y ejes centrados en el origen
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_aspect('equal', 'box')
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

# Datos iniciales
newen = BRW.BRW_IDLA_PERC(p)
newen.crear_perc(100,0.4,0.4)
print("Perc generada, inciando r inicial")

for i in range(r_inicial-1): # por el pi...

    newen.crear_particula()
    while not newen.vacio: # ACTUALIZAMOS HASTA QUEDAR si particula 
        newen.actualizar() # movemos 

# Creamos una más por motivos de animacion que si no se me rompe la webada de abajo
newen.crear_particula()

N = 100 # NADIE SABE PARA QUE ES ESTE N

# Preparar visual
cluster = ax.scatter(newen.mapa[:, 0], newen.mapa[:,1], s=20, color='red')
scat = ax.scatter(newen.particulas[:, 0], newen.particulas[:, 1], s=20, color='black', alpha=0.3)
texto = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top')

"""
QUE se debe hacer, ? 😭

"""

# Función que actualiza el gráfico en cada frame
def update(frame): # falta trabajar bastante esta parte
    # TO DOS: 
    # parar simulacion (caso sub critico) (averiguar intervencion del usuario para recontinuar(?))
    # simulacion GRANDE
    # ver cuna
    global newen
    # Movimiento aleatorio pequeño
    newen.actualizar()
    if newen.vacio:
        # sos un boludo, si mueree la simulacion el siguiente update pide actualizar particulas vacias tenes que hacer un break
        print("⚠️  Evento detectado, simulación pausada.")
        #input("Presiona Enter para continuar...")
        print("Reanudando...")
        newen.crear_particula()
        return cluster, scat, texto

    x, y = zip(*newen.particulas)
    cluster.set_offsets(np.c_[newen.mapa[:,0], newen.mapa[:,1]])
    scat.set_offsets(np.c_[x, y])
    texto.set_text(f'Partículas activas: {len(newen.particulas)}\n(a, b) = {newen.obtener_parametros()}')

    return cluster, scat, texto

# Crear la animación
ani = FuncAnimation(fig, update, interval=1, blit=True)

plt.show()






