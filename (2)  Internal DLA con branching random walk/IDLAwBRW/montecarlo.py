'''Vamos a intentar estimar los parámetros a y b de la elipse
 generada por el crecimiento del cluster bajo percolaciones aleatorias'''

import BRW
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

"""Idea 1: si los parametros a y b de la elipse son un factor que se escala
entoncces se tiene que Area = \pi (a*S) (b*S)
=> S = sqrt(Area/abpi)
Luego podemos ver como varía esta escala dependiendo de los parametros de descarte horizontal y vertical de la percolacion"""

def generar_resultados():
    resultados = []

    for N in [1000, 2000, 3000, 4000]:
        for p_hor in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for p_ver in np.linspace(0,0.9-p_hor,4):
                newen = BRW.BRW_IDLA_PERC(0.0) # Creamos un idla con percolacion SIN BRANCHING
                newen.crear_perc(100, p_ver, p_hor)

                # Hacer crecer el cluster hasta tamaño N (esto deberia ser un metodo interno de brw u.u)
                print("Empezando simulacion (N,ph,pv)", N, p_hor, p_ver)
                while newen.N < N:
                    if newen.vacio:
                        newen.crear_particula()
                    newen.actualizar()

                # Ya llegamos al punto en el que queriamos estar ahora registramos a y b
                
                a, b = newen.obtener_parametros()
                S = N / (a*b*np.pi)
                resultados.append({
                    "N": N,
                    "p_hor": p_hor,
                    "p_ver": p_ver,
                    "a": a,
                    "b": b,
                    "S": S })
                
    return resultados


def main():    

    resultados = generar_resultados()
    with open("resultados.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

def segundomain():
    # 1) Cargar el CSV
    df = pd.read_csv("resultados.csv")

    # 2) Obtener niveles únicos de p_hor y p_ver
    p_hors = sorted(df["p_hor"].unique())
    p_vers = sorted(df["p_ver"].unique())

    # 3) Crear matriz S (heatmap)
    Z = np.zeros((len(p_vers), len(p_hors)))

    for i, pver in enumerate(p_vers):
        for j, phor in enumerate(p_hors):
            fila = df[(df["p_ver"] == pver) & (df["p_hor"] == phor)]
            if not fila.empty:
                Z[i, j] = fila["S"].values[0]
            else:
                Z[i, j] = np.nan   # por si falta algún valor

    # 4) Dibujar heatmap
    plt.figure(figsize=(8,6))
    plt.pcolormesh(p_hors, p_vers, Z, shading="auto")
    plt.colorbar(label="S")
    plt.xlabel("p_hor")
    plt.ylabel("p_ver")
    plt.title("Heatmap de S según p_hor y p_ver")
    plt.tight_layout()
    plt.show()

def tercermain():
   
    
    # ============================
    # 1) Tu función S_teórica
    # ============================
    def S_teorica(ph, pv):
        # EJEMPLO: tú cambias esta función
        # Supongamos S depende linealmente:
        #return (1-ph)*(1-pv)/(pv+ph)**2
        #return 0.5+(1-ph)*(1-pv) # Masomenos pero muy abajo le falta algo como un + C ( no tiene sentido?)
        #return np.exp(-(ph**2 +pv**2)) #masmenos, bueno igual peroinexplicable
        return (pv+ph)**2/((1-pv)*(1-ph)) # Proporcional al utilizado
        
        return 2*(pv+ph)**2/(ph*pv-1)
        # Cámbiala por tu fórmula experimental


    # ============================
    # 2) Cargar datos
    # ============================
    df = pd.read_csv("resultados.csv")

    p_hor_vals = df["p_hor"].values
    p_ver_vals = df["p_ver"].values
    S_exp = df["S"].values


    # ============================
    # 3) Calcular S_teo y distancia cuadrada total
    # ============================
    S_teo = S_teorica(p_hor_vals, p_ver_vals)
    dist2_total = np.sum((S_exp - S_teo)**2)

    print("Distancia cuadrada total:")
    print(dist2_total)


    # ============================
    # 4) Graficar resultados en 3D
    # ============================
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111, projection='3d')

    # Puntos experimentales
    p = ax.scatter(
        p_hor_vals,
        p_ver_vals,
        S_exp,
        c=df["N"],
        cmap="viridis",
        s=60,
        label="Datos simulados"
    )

    # Superficie teórica (wireframe)
    ph = np.linspace(min(p_hor_vals), max(p_hor_vals), 30)
    pv = np.linspace(min(p_ver_vals), max(p_ver_vals), 30)
    PH, PV = np.meshgrid(ph, pv)
    Ssurf = S_teorica(PH, PV)

    ax.plot_wireframe(PH, PV, Ssurf, color="red", linewidth=1, label="S_teórica")

    # Etiquetas
    ax.set_xlabel("p_hor")
    ax.set_ylabel("p_ver")
    ax.set_zlabel("S")
    ax.set_title(f"Comparación datos vs función teórica\nDist^2 total = {dist2_total:.3f}")

    # Leyenda + barra de color
    plt.colorbar(p, ax=ax, label="N")
    plt.tight_layout()
    plt.show()


    pass
if __name__ == '__main__':
    #main()
    tercermain()
