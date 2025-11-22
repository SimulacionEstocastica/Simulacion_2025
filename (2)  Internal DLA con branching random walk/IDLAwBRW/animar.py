import BRW

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse




def pedir_float(msg):
    while True:
        s = input(msg).strip()
        
        if s == "":
            return 0
        
        try:
            x = float(s)
            if x < 0 or x > 1:
                raise ValueError("Input debe estar entre 0 y 1")
            return x
        except ValueError:
            print("Entrada inválida, intenta nuevamente.")
def pedir_int(msg):
    while True:
        try:
            return int(input(msg))
        except ValueError:
            print("Entrada inválida, intenta nuevamente.")
# SETUP

N_inicial = pedir_int("Ingresa tamaño inicial del cluster: ") # NO ES UN RADIO
""" Se usa para establecer una cantidad inicial de partículas antes de la animación

"""
p = pedir_float("Ingresa prob. de apareamiento: ")
""" Probabilidad de aparearse 
"""

p_hor = pedir_float("Ingresa prob. descarte horizontal: ")


p_ver = pedir_float("Ingresa prob. descarte vertical: ")

""" Probabilidades de descartar un camino horizontal o vertical en la percolacion"""

pasos = pedir_int("Cuanto debe crecer el cluster por frame: ")

FPS = pedir_int("imágenes por segundo: ")
frames = pedir_int("Cantidad total de frames : ")

nombre = input("NOMBRE ARCHIVO: ").strip() # nombre se guarda sin espacios?


#a, b =  (1-p_ver), (1-p_hor
# esto no a, b =  1/(1-p_hor) , 1/(1-p_ver)
#a, b = (1-p_ver)/( p_ver+ p_hor), (1-p_hor)/( p_ver+ p_hor) # esto un poco para valores bajos de ps
C = 1
if p_ver + p_hor == 0:
    a, b= 1, 1
else:
    a, b = C*(1-p_ver)/( p_ver+ p_hor), C*(1-p_hor)/( p_ver+ p_hor) 
'''Conjetura sobre como se comporta el crecimiento del cluster como una ellipse,'''

N = min(100,max(20, (3*N_inicial+9)**0.5)).__int__() # N aproximado para ver de que tamaño hacer el dibujo (y la perc)?

# Datos iniciales
newen = BRW.BRW_IDLA_PERC(p)
newen.crear_perc(N, p_ver, p_hor)

print("Iniciando simulacion inicial!")
while newen.N < N_inicial:
    # creamos una particula y esperamos a q muera
    if newen.N % 1000 == 0:
        print(newen.N, "particulas en el cluster") # esto a veces (cuando hay branchin) no se imprime
    newen.crear_particula()
    while not newen.vacio and newen.N < N_inicial: 
        newen.actualizar()



# Creamos una más por motivos de animacion que si no se me rompe la webada de abajo
if newen.vacio:
    newen.crear_particula()


print("iniciande dibuje")
# Crear figura y ejes centrados en el origen
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-N,N)
ax.set_ylim(-N, N)
ax.set_aspect('equal', 'box')
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

# Preparar visual
ellipse = Ellipse((0, 0), width=2*a, height=2*b, fill=False, color='indigo')
ax.add_patch(ellipse)
scat = ax.scatter(newen.particulas[:, 0], newen.particulas[:, 1], s=20, color='black', zorder=2, alpha = 0.5)
cluster = ax.scatter(newen.mapa[:, 0], newen.mapa[:,1], s=20, color='darkviolet', zorder= -1)
texto = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', zorder=10)

"""
QUE se debe hacer, ? 😭
TODO
 - IMPLEMENTAR VALIDAR PERCOLACION  
 - input colores?
 - elegir "tipos de dibujo?" se va a implementar un update2() para dibujar la primera parte
"""

# Función que actualiza el gráfico en cada frame
def update(frame):

    global newen # nuestra simulacion
    
    # Tamaño del cluster "|A(n)|"   
    # Movemos hasta actualizar el cluster??

    N = newen.N
    objetivo = N + pasos   

    while newen.N < objetivo:
        newen.actualizar() #ATENCION ESTO SE ROMPE CON P =! DE PYCLE 0
        if newen.vacio:
            newen.crear_particula()

        
    N = newen.N

    # Calcular tamaño del     
    scale =  (np.pi/0.69)**0.5 * (N / (np.pi* a*b))**0.5 


    # Actualizar ovalo
    ellipse.width = a*scale
    ellipse.height = b*scale



    x, y = zip(*newen.particulas)
    cluster.set_offsets(np.c_[newen.mapa[:,0], newen.mapa[:,1]])
    scat.set_offsets(np.c_[x, y])
    texto.set_text(f'Tamaño cluster: {N}\n $p$ branch = {p}\n($p_h$,  $p_v$) : ({p_ver}, {p_hor})\nPartículas activas: {len(newen.particulas)}')

    return cluster, scat, texto, ellipse

def update2(frame):
    """ 
    Version alternativa
    Editar a gusto es una mierda
    Un frame cada simulació"""
    global newen # nuestra simulacion
    
    # Tamaño del cluster "|A(n)|"
    # Movemos hasta actualizar el cluster??

    newen.actualizar()
    if newen.vacio:
        newen.crear_particula()
        
    N = newen.N

    # Calcular tamaño del     
    scale =  (np.pi/0.69)**0.5 * (N / (np.pi* a*b))**0.5 


    # Actualizar ovalo
    ellipse.width = a*scale
    ellipse.height = b*scale


    x, y = zip(*newen.particulas)
    cluster.set_offsets(np.c_[newen.mapa[:,0], newen.mapa[:,1]])
    scat.set_offsets(np.c_[x, y])
    texto.set_text(f'Tamaño cluster: {N}\n ')#$p$ branch = {p}\n($p_h$,  $p_v$) : ({p_ver}, {p_hor})\nPartículas activas: {len(newen.particulas)}')

    return scat, cluster#, texto#, ellipse

# Crear la animación
ani = FuncAnimation(fig, update, frames, interval=10, blit=True)
#plt.show()
from matplotlib.animation import PillowWriter
print("guardando animatsion")
ani.save(f"gifs/{nombre}.gif", writer=PillowWriter(fps=FPS))
print("animatsion gualdada")



