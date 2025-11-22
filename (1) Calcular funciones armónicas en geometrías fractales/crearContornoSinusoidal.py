# Todo está hecho por gemini, solo fue debugeado

import numpy as np
import matplotlib.pyplot as plt
import csv

def generar_contorno_sinusoidal_arreglado(archivo_frontera_csv):
    """
    Carga la matriz de frontera y genera valores sinusoidales en el contorno.
    Asegura que el ángulo (theta) esté en el rango [0, 2*pi] para garantizar
    el número correcto de ciclos.
    """
    
    # --- PARÁMETROS DE CONFIGURACIÓN ---
    # AJUSTA ESTOS VALORES SEGÚN TU NECESIDAD
    VALOR_DE_FRONTERA = 1   # Valor que marca la frontera circular en tu CSV.
    AMPLITUD = 10.0         # Amplitud (A) de la onda sinusoidal.
    
    # 🌟 ¡AJUSTA ESTE VALOR PARA VER MÁS O MENOS PICOS/VALLES! 🌟
    CICLOS_SEN = 5          # Número de ciclos (k) completos en la frontera.
    
    # --- 1. CARGAR MATRIZ DE FRONTERA ---
    try:
        F = np.genfromtxt(archivo_frontera_csv, delimiter=',', dtype=np.int8)
    except FileNotFoundError:
        print(f"❌ Error: Archivo de frontera '{archivo_frontera_csv}' no encontrado.")
        return
    
    N = F.shape[0]
    if N != F.shape[1]:
         print(f"❌ Error: La matriz de frontera no es cuadrada ({F.shape}).")
         return
    print(f"✅ Matriz Frontera (F) cargada. Tamaño NxN: {N}x{N}")

    # --- 2. CALCULAR COORDENADAS DEL CENTRO Y ÁNGULOS (AJUSTE CLAVE AQUÍ) ---
    
    centro = N / 2.0
    grilla_coords = np.arange(N)
    X, Y = np.meshgrid(grilla_coords, grilla_coords)
    
    # Coordenadas relativas al centro
    X_rel = X - centro
    Y_rel = Y - centro
    
    # 2.2. Calcular el ángulo (theta) y asegurar el rango [0, 2*pi]
    # np.arctan2 da [-pi, pi]. np.mod(theta, 2*np.pi) convierte ese rango a [0, 2*pi].
    theta = np.arctan2(Y_rel, X_rel)
    theta = np.mod(theta, 2 * np.pi) 
    
    print("✅ Ángulos (theta) calculados en rango [0, 2*pi].")
    
    # --- 3. GENERAR VALORES SINUSOIDALES ---
    
    # Z = A * sin(theta * k)
    # Al usar el rango [0, 2*pi], el argumento (theta * k) ahora va de 0 a (k * 2*pi), 
    # garantizando exactamente 'k' ciclos completos de la función seno.
    Z_sinusoidal = AMPLITUD * np.sin(theta * CICLOS_SEN)

    # 3.2. Aplicar la máscara de frontera
    Z_contorno = np.zeros((N, N), dtype=np.float64)
    VALOR_DE_FRONTERA_FOUND = np.unique(F[F != 0])
    
    if VALOR_DE_FRONTERA_FOUND.size > 0:
        # Usamos el valor encontrado en la matriz F (si no es 0)
        VALOR_DE_FRONTERA = VALOR_DE_FRONTERA_FOUND[0]
    else:
        # Si no se encuentra nada, asumimos el valor por defecto 1 (o ajusta manualmente)
        pass 
    
    Z_contorno[F == VALOR_DE_FRONTERA] = Z_sinusoidal[F == VALOR_DE_FRONTERA]
    
    print("✅ Valores sinusoidales generados y aplicados solo en el contorno.")
    
    # --- 4. GUARDAR Y VISUALIZAR ---

    nombre_csv_salida = f"contorno_sinusoidal_{N}x{N}_k{CICLOS_SEN}.txt"
    np.savetxt(
        nombre_csv_salida, 
        Z_contorno, 
        delimiter=',', 
        fmt='%.6f' # Formato flotante para precisión
    )
    print(f"\n💾 Contorno sinusoidal guardado en: {nombre_csv_salida}")

    # Visualización
    plt.figure(figsize=(8, 8))
    plt.imshow(Z_contorno, cmap='seismic', origin='lower')
    plt.colorbar(label='Valor de Contorno Z')
    plt.title(f'Contorno Sinusoidal (A={AMPLITUD}, k={CICLOS_SEN})')
    plt.axis('off')
    plt.show()

    return Z_contorno

# --- EJECUCIÓN ---
# REEMPLAZA ESTE NOMBRE DE ARCHIVO
archivo_frontera = input("Nombre del archivo CSV de frontera circular: ")

generar_contorno_sinusoidal_arreglado(archivo_frontera)
