# play_vs_ia.py
import torch
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device.upper()}")

from hex_game import HexGame
from hex_network import create_hex_network
from hex_mcts import MCTS, select_action

def cargar_modelo(ruta_modelo):
    print(f"Cargando modelo: {ruta_modelo}")
    model = create_hex_network(device=device)
    
    try:
        checkpoint = torch.load(ruta_modelo, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(ruta_modelo, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Modelo cargado correctamente")
    return model

def mostrar_tablero(juego):
    print("\n" + "="*40)
    print("Tablero actual:")
    print(juego.board)
    print(f"Coordenadas: (0,0) a ({juego.size-1},{juego.size-1})")

def obtener_movimiento_humano(juego):
    while True:
        try:
            entrada = input("\nTu movimiento (fila columna o 'swap'): ").strip()
            
            if entrada.lower() == 'swap':
                if 'swap' in juego.get_legal_moves():
                    return 'swap'
                else:
                    print(" Swap no permitido ahora")
                    continue
            
            partes = entrada.split()
            if len(partes) != 2:
                print("Usa: fila columna (ej: 3 4)")
                continue
                
            fila, col = int(partes[0]), int(partes[1])
            
            if not (0 <= fila < juego.size and 0 <= col < juego.size):
                print(f"Usa números entre 0 y {juego.size-1}")
                continue
            
            movimiento = (fila, col)
            
            if movimiento in juego.get_legal_moves():
                return movimiento
            else:
                print("Casilla ocupada o movimiento ilegal")
                
        except ValueError:
            print("Ingresa números válidos")
        except KeyboardInterrupt:
            return None

def jugar_contra_ia(ruta_modelo, humano_primero=True):
    modelo = cargar_modelo(ruta_modelo)
    juego = HexGame(size=7)
    
    print("\n🎮 HEX - Humano vs IA")
    print("="*50)
    print("REGLAS:")
    print("- Tú: X (Azul) - Conecta IZQUIERDA-DERECHA")
    print("- IA: O (Rojo) - Conecta ARRIBA-ABAJO")
    print("- Swap: En el primer turno del segundo jugador")
    print("- Movimientos: 'fila columna' (ej: 3 4)")
    print()
    
    juego.reset()
    
    # El humano es el jugador 1 si juega primero, sino -1
    humano_es = 1 if humano_primero else -1
    
    while True:
        mostrar_tablero(juego)
        
        # Determinar de quién es el turno
        es_turno_humano = (juego.current_player == humano_es)
        color_actual = 'X - Azul' if juego.current_player == 1 else 'O - Rojo'
        
        if es_turno_humano:
            print(f"\nTu turno ({color_actual})")
            movimiento = obtener_movimiento_humano(juego)
            if movimiento is None:
                return

            juego.apply_move(movimiento)
            
            if movimiento == 'swap':
                print("✅ Has usado SWAP")
                # Después del swap, el turno cambia automáticamente en apply_move
            else:
                print(f"✅ Jugaste: {movimiento}")

        else:
            print(f"\nTurno de la IA ({color_actual})")
            print(" IA pensando...")
            
            from hex_env import HexEnvironment
            env_temp = HexEnvironment(juego.size)
            env_temp.game = juego.clone()
            
            mcts = MCTS(modelo, env_temp, num_simulations=50, device=device)
            estado = env_temp.state_to_tensor()
            mascara_legal = env_temp.get_legal_actions_mask()
            
            probs_acciones = mcts.run(estado, mascara_legal)
            accion = select_action(probs_acciones, temperature=0.1)
            
            if accion == juego.size * juego.size:
                movimiento = 'swap'
            else:
                fila = accion // juego.size
                col = accion % juego.size
                movimiento = (fila, col)
            
            juego.apply_move(movimiento)
            
            if movimiento == 'swap':
                print("IA usó SWAP")
            else:
                print(f"IA jugó: {movimiento}")
        
        terminado, ganador = juego.is_terminal()
        if terminado:
            print("\n" + "- " * 10)
            print("¡JUEGO TERMINADO!")
            mostrar_tablero(juego)
            
            # Determinar quién ganó
            if ganador == humano_es:
                print("¡GANASTE!")
            else:
                print("La IA ganó")
                
            print("-" * 10)
            break

def main():
    ruta_modelo = "checkpoints/hex_model_epoch_100_final.pth" # <- AQUI SE PUEDE ESCOGER CON CUANTA CANTIDAD DE ENTRENAMIENTO JUGAR
    
    if not os.path.exists(ruta_modelo):
        print(f"No se encontró el modelo: {ruta_modelo}")
        return
    
    while True:
        print("\nHEX - Jugar contra IA")
        print("1. Yo primero (X - Azul)")
        print("2. IA primero (O - Rojo)") 
        print("3. Salir")
        
        opcion = input("\nElige (1-3): ").strip()
        
        if opcion == '1':
            jugar_contra_ia(ruta_modelo, humano_primero=True)
        elif opcion == '2':
            jugar_contra_ia(ruta_modelo, humano_primero=False)
        elif opcion == '3':
            print("¡Hasta luego!")
            break
        else:
            print("Opción inválida")

if __name__ == "__main__":
    main()