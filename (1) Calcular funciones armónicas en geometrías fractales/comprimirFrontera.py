# Todo está hecho por gemini, solo fue debugeado

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def redimensionar_y_visualizar_fractal():
    """
    Función principal: Solicita el nombre del archivo CSV y el nuevo tamaño,
    luego carga, redimensiona, binariza y guarda el resultado junto con imágenes
    visuales del antes y después.
    """
    
    # --- 1. SOLICITAR DATOS DE ENTRADA ---
    try:
        archivo_csv = input("Nombre del archivo CSV de la frontera (e.g., fractal_5001x5001.csv): ")
        # Intentamos obtener el tamaño de un número entero positivo
        tam = int(input("Nuevo tamaño del lado (e.g., 501): "))
        
        if tam <= 0:
            print("❌ Error: El tamaño debe ser un entero positivo.")
            return

    except ValueError:
        print("❌ Error: El tamaño debe ser un número entero.")
        return

    # --- 2. CARGAR LA MATRIZ ---
    print(f"\nCargando '{archivo_csv}'...")
    try:
        data_original = np.genfromtxt(
            archivo_csv, 
            delimiter=',', 
            dtype=np.float32 
        )
    except FileNotFoundError:
        print(f"❌ Error: El archivo '{archivo_csv}' no fue encontrado. Verifica el nombre.")
        return 
    except Exception as e:
        print(f"❌ Error al leer el CSV: {e}")
        return

    original_shape = data_original.shape
    if len(original_shape) != 2 or original_shape[0] != original_shape[1]:
         print(f"❌ Error: Se esperaba una matriz cuadrada 2D. Se encontró: {original_shape}")
         return
         
    original_size = original_shape[0]
    print(f"✅ Matriz cargada. Forma original: {original_shape}")
    
    # --- 3. REDIMENSIONAR (ZOOM) y BINARIZAR ---
    
    # Calcular factor de zoom
    factor_zoom = tam / original_size
    
    # Redimensionar usando interpolación bicúbica (order=3)
    data_reducida_float = zoom(data_original, factor_zoom, order=3)

    # Binarizar el resultado (aplicar umbral de 0.5)
    data_reducida_binaria = (data_reducida_float >= 0.5).astype(np.int8)
    
    reducida_shape = data_reducida_binaria.shape
    print(f"✅ Redimensionado completado. Forma final: {reducida_shape}")
    
    # --- 4. GUARDAR RESULTADOS ---
    
    ## A. Guardar el nuevo archivo CSV 
    nombre_csv_salida = f"fractal_reducido_{reducida_shape[0]}x{reducida_shape[1]}.txt"
    np.savetxt(
        nombre_csv_salida, 
        data_reducida_binaria, 
        delimiter=',', 
        fmt='%d' 
    )
    print(f"\n💾 Matriz reducida guardada en: {nombre_csv_salida}")
    
    ## B. Visualizar y Guardar Imágenes
    
    # Crear una figura con dos subgráficos (antes y después)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Imagen Original
    axes[0].imshow(data_original, cmap='binary') 
    axes[0].set_title(f"Original ({original_size}x{original_size})")
    axes[0].axis('off')

    # Imagen Reducida
    axes[1].imshow(data_reducida_binaria, cmap='binary')
    axes[1].set_title(f"Reducida y Binarizada ({reducida_shape[0]}x{reducida_shape[1]})")
    axes[1].axis('off')

    # Guardar la figura completa
    nombre_img_comparacion = f"comparacion_fractal_{original_size}_a_{reducida_shape[0]}.png"
    plt.tight_layout()
    plt.savefig(nombre_img_comparacion)
    plt.close(fig) # Cierra la figura para liberar memoria
    print(f"🖼️ Imagen de comparación guardada en: {nombre_img_comparacion}")

# --- Ejecución del Script ---
if __name__ == "__main__":
    redimensionar_y_visualizar_fractal()