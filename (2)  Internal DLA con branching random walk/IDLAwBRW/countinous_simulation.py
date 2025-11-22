import BRW
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
""" DEPRECATED, archivo principal: BRW
Herramienta de animacion: animar.py"""
# VAARIABLES
p = 0.4
lamba = 10


# Crear figura y ejes
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal', 'box')
ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

# Inicializar proceso
benja = BRW.BRW_IDLA(p)
benja.crear_particula()

# Parámetros del proceso de Poisson
tasa_movimiento = lamba  # λ - eventos por segundo

# Variables para control de tiempo real
ultimo_tiempo = time.time()
tiempo_simulacion = 0.0
tiempo_acumulado = 0.0
proceso_terminado = False

# Configurar gráficos
cluster = ax.scatter(benja.mapa[:, 0], benja.mapa[:, 1], s=20, color='red')
scat = ax.scatter(benja.particulas[:, 0], benja.particulas[:, 1], s=20, color='black', alpha=0.3)
tiempo_texto = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top')

# Texto para indicar que el proceso terminado (arriba a la derecha, más pequeño)
terminado_texto = ax.text(0.98, 0.98, 'PROCESO TERMINADO', 
                         transform=ax.transAxes, fontsize=10, color='red', 
                         fontweight='bold', ha='right', va='top',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
terminado_texto.set_visible(False)

def update(frame):
    global benja, ultimo_tiempo, tiempo_simulacion, tiempo_acumulado, proceso_terminado
    
    # Si el proceso ya terminó, no hacer nada
    if proceso_terminado:
        return cluster, scat, tiempo_texto, terminado_texto
    
    # Calcular delta tiempo REAL desde el último frame
    tiempo_actual = time.time()
    dt_real = tiempo_actual - ultimo_tiempo
    ultimo_tiempo = tiempo_actual
    
    # Acumular tiempo de simulación
    tiempo_simulacion += dt_real
    tiempo_acumulado += dt_real
    
    # Verificar si no hay partículas
    if len(benja.particulas) == 0:
        proceso_terminado = True
        terminado_texto.set_visible(True)
        tiempo_texto.set_text(f'Tiempo final: {tiempo_simulacion:.2f}s\nPROCESO TERMINADO')
        return cluster, scat, tiempo_texto, terminado_texto
    
    # Calcular número esperado de eventos en este intervalo de tiempo real
    eventos_esperados = tasa_movimiento * dt_real
    
    # Generar número de eventos usando distribución de Poisson
    num_eventos = np.random.poisson(eventos_esperados)
    
    # Ejecutar los eventos
    for _ in range(num_eventos):
        if len(benja.particulas) > 0:
            benja.actualizar()
        else:
            # Si durante la ejecución se acaban las partículas
            proceso_terminado = True
            terminado_texto.set_visible(True)
            break
    
    # Actualizar visualización
    cluster.set_offsets(np.c_[benja.mapa[:, 0], benja.mapa[:, 1]])
    if len(benja.particulas) > 0:
        scat.set_offsets(np.c_[benja.particulas[:, 0], benja.particulas[:, 1]])
    
    # Actualizar texto con información de tiempo
    fps = 1.0 / dt_real if dt_real > 0 else 0
    tiempo_texto.set_text(f'Tiempo: {tiempo_simulacion:.2f}s\n'
                         f'Eventos: {num_eventos}\n'
                         f'FPS: {fps:.1f}\n'
                         f'Partículas: {len(benja.particulas)}')
    
    return cluster, scat, tiempo_texto, terminado_texto

# Configurar animación - usar intervalo mínimo para tiempo real
ani = FuncAnimation(fig, update, frames=None, interval=1, blit=True, cache_frame_data=False)

plt.show()