# Todo está hecho por gemini, solo fue debugeado

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import cmath

def graficar_campo_vectorial_complejo_doble():
    """
    Grafica la Parte Real y la Parte Imaginaria de un campo vectorial complejo
    lado a lado en una grilla uniforme, usando una máscara para excluir puntos.
    """
    
    # --- 1. SOLICITAR NOMBRES DE ARCHIVO ---
    print("--- Configuración de Archivos ---")
    archivo_valores = input("Nombre del archivo CSV de VALORES (Campo Vectorial Complejo): ")
    archivo_frontera = input("Nombre del archivo CSV de FRONTERA (Máscara binaria 0/1): ")
    print("-" * 30)

    # --- 2. CARGAR MATRIZ DE FRONTERA (MÁSCARA) ---
    try:
        F = np.genfromtxt(archivo_frontera, delimiter=',', dtype=np.int8)
    except FileNotFoundError:
        print(f"❌ Error: Archivo de frontera '{archivo_frontera}' no encontrado.")
        return
    
    N = F.shape[0]
    if N != F.shape[1]:
         print(f"❌ Error: La matriz de frontera no es cuadrada ({F.shape}).")
         return
    print(f"✅ Matriz Frontera (F) cargada. Tamaño NxN: {N}x{N}")

    # --- 3. CARGAR Y PARSEAR MATRIZ DE VALORES ---
    complejos_flat = []
    try:
        with open(archivo_valores, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for element_str in row:
                    if element_str.strip():
                        try:
                            parsed_str = element_str.replace('i', 'j')
                            if parsed_str.strip() == 'j': parsed_str = '1j'
                            elif parsed_str.strip() == '-j': parsed_str = '-1j'
                            cnum = complex(parsed_str)
                            complejos_flat.append(cnum)
                        except ValueError as e:
                            print(f"❌ Error de parsing: No se pudo convertir '{element_str}' a complejo. Error: {e}")
                            return
    except FileNotFoundError:
        print(f"❌ Error: Archivo de valores '{archivo_valores}' no encontrado.")
        return
    
    if len(complejos_flat) != N * N:
        print(f"❌ Error: El número de valores complejos parseados ({len(complejos_flat)}) no coincide con el tamaño esperado ({N*N}).")
        return

    complejos_array = np.array(complejos_flat)
    
    # Separar Partes Real e Imaginaria
    Z_real_valores = complejos_array.real.reshape((N, N))
    Z_imag_valores = complejos_array.imag.reshape((N, N))
    
    # --- 4. GENERAR MATRIZ DE GRILLA UNIFORME (X, Y) ---
    rango_grilla = np.arange(N)
    X, Y = np.meshgrid(rango_grilla, rango_grilla)
    
    # --- 5. APLICAR MÁSCARA A AMBAS PARTES ---
    F_mask = (F == 1)
    
    Z_real_grafica = Z_real_valores.astype(np.float64).copy()
    Z_real_grafica[F_mask] = np.nan

    Z_imag_grafica = Z_imag_valores.astype(np.float64).copy()
    Z_imag_grafica[F_mask] = np.nan
    
    print("✅ Máscara aplicada a Partes Real e Imaginaria.")

    # --- 6. GRAFICAR SUPERFICIE 3D LADO A LADO ---
    
    # Definir el stride (paso)
    stride_val = max(1, N // 100) 
    if N <= 101:
        stride_val = 1
        
    # Crear una figura ancha para los dos gráficos
    fig_3d = plt.figure(figsize=(24, 10))
    
    # ---------------------------------
    # SUBPLOT 1: PARTE REAL
    # ---------------------------------
    ax_real = fig_3d.add_subplot(1, 2, 1, projection='3d') # (Filas, Columnas, Índice)
    surface_real = ax_real.plot_surface(
        X, Y, Z_real_grafica, 
        cmap='hsv', # Color cíclico (Rojo a Rojo)
        rstride=stride_val, 
        cstride=stride_val,
        linewidth=0, 
        antialiased=False, 
        alpha=1.0 
    )
    
    ax_real.set_title(f'3D: Parte Real ({N}x{N})')
    ax_real.set_xlabel('X (Columna)')
    ax_real.set_ylabel('Y (Fila)')
    ax_real.set_zlabel('Z (Parte Real)')
    fig_3d.colorbar(surface_real, ax=ax_real, shrink=0.7, label='Parte Real')
    
    # ---------------------------------
    # SUBPLOT 2: PARTE IMAGINARIA
    # ---------------------------------
    ax_imag = fig_3d.add_subplot(1, 2, 2, projection='3d') 
    surface_imag = ax_imag.plot_surface(
        X, Y, Z_imag_grafica, 
        cmap='hsv', 
        rstride=stride_val, 
        cstride=stride_val,
        linewidth=0, 
        antialiased=False, 
        alpha=1.0 
    )
    
    ax_imag.set_title(f'3D: Parte Imaginaria ({N}x{N})')
    ax_imag.set_xlabel('X (Columna)')
    ax_imag.set_ylabel('Y (Fila)')
    ax_imag.set_zlabel('Z (Parte Imaginaria)')
    fig_3d.colorbar(surface_imag, ax=ax_imag, shrink=0.7, label='Parte Imaginaria')
    
    plt.tight_layout() # Ajusta el espaciado para que no se superpongan
    plt.show() # Muestra la figura con ambos gráficos interactivos

    # --- 7. GENERAR Y GUARDAR MAPAS 2D (VISTA DESDE ARRIBA) ---
    print("\nGenerando y guardando mapas 2D (vista desde arriba)...")
    
    data_to_save = [
        (Z_real_grafica, "Parte Real", "real"), 
        (Z_imag_grafica, "Parte Imaginaria", "imaginaria")
    ]
    
    for Z_data, title, file_tag in data_to_save:
        fig_2d, ax_2d = plt.subplots(figsize=(10, 10))
        
        im = ax_2d.imshow(Z_data, cmap='hsv', origin='lower')
        
        fig_2d.colorbar(im, ax=ax_2d, label=f'Z ({title})')
        
        nombre_mapa_2d = f"mapa_2d_campo_{file_tag}_{N}x{N}.png"
        ax_2d.set_title(f'Mapa 2D: Campo ({title}) sobre Grilla Uniforme ({N}x{N})')
        ax_2d.set_xlabel('Índice de Columna X')
        ax_2d.set_ylabel('Índice de Fila Y')
        
        plt.savefig(nombre_mapa_2d, bbox_inches='tight', dpi=300)
        plt.close(fig_2d)
        print(f"✅ Mapa 2D de {title} guardado en: {nombre_mapa_2d}")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    graficar_campo_vectorial_complejo_doble()