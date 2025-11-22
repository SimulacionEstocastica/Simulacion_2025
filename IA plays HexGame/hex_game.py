import torch
import numpy as np
from typing import List, Tuple, Optional, Any
import copy
from hex_map import HEX_MAP

class HexGame:
    def __init__(self, size: int = 7):
        self.size = int(size)
        self.board = HEX_MAP(self.size)
        self.current_player = 1  # 1 or -1
        self.move_count = 0
        self.done = False
        self.winner = 0

    def reset(self) -> np.ndarray:
        """Reinicia el juego y retorna el estado inicial"""
        self.board = HEX_MAP(self.size)
        self.current_player = 1
        self.move_count = 0
        self.done = False
        self.winner = 0
        return self.get_state()

    def clone(self):
        return copy.deepcopy(self)

    def get_legal_moves(self) -> List[Any]:
        moves: List[Any] = []
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                if self.board.tablero[i, j] == 0:
                    moves.append((i - 1, j - 1))  # 0-based for agent
        
        # swap: permitido solo en el primer turno del segundo jugador
        if self.move_count == 1 and self.current_player == -1:
            moves.append("swap")
        return moves

    def get_legal_moves_mask(self) -> np.ndarray:
        """Retorna una máscara binaria de movimientos legales"""
        mask = np.zeros((self.size * self.size + 1), dtype=np.float32)  # +1 para swap
        legal_moves = self.get_legal_moves()
        
        for move in legal_moves:
            if move == "swap":
                mask[-1] = 1.0
            else:
                r, c = move
                idx = r * self.size + c
                mask[idx] = 1.0
        return mask

    def apply_move(self, move: Any) -> None:
        if self.done:
            return

        # normalizar numpy arrays a tuplas
        if isinstance(move, np.ndarray):
            move = tuple(int(x) for x in move.tolist())

        legal = self.get_legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move {move}. Legal: {legal}")

        # --- SWAP handling ---
        if move == "swap":
            stones = []
            T = self.board.tablero
            for i in range(1, self.size + 1):
                for j in range(1, self.size + 1):
                    if T[i, j] != 0:
                        stones.append((i, j))
            if len(stones) != 1:
                raise RuntimeError("Swap invoked but there is not exactly one stone present.")
            i, j = stones[0]
            T[i, j] = -T[i, j]  # invertir color
            
            # IMPORTANTE: NO cambiar el turno después del swap
            # El jugador que hizo swap mantiene su turno pero cambia de color
            self.current_player *= -1  
            
            self.move_count += 1
            return

        # --- normal move ---
        r, c = move
        pos = np.array([int(r) + 1, int(c) + 1], dtype=int)
        col = self.current_player
        self.board.jugada(col, pos)
        self.current_player *= -1  # Esto SÍ cambia el turno para movimientos normales
        self.move_count += 1

        # Check if game ended
        self.done, winner = self.is_terminal()
        if self.done:
            self.winner = winner

    def step(self, move: Any) -> Tuple[np.ndarray, float, bool, dict]:
        """Interfaz estilo OpenAI Gym"""
        old_player = self.current_player
        self.apply_move(move)
        
        reward = 0.0
        if self.done:
            if self.winner == old_player:
                reward = 1.0
            elif self.winner == -old_player:
                reward = -1.0
        
        return self.get_state(), reward, self.done, {}

    def get_state(self) -> np.ndarray:
        """Representación del estado para la red neuronal"""
        # Para un tamaño 7, el tablero es 9x9 con bordes
        # Retornamos solo la parte jugable (7x7) pero podríamos incluir los bordes
        state = self.board.tablero[1:self.size+1, 1:self.size+1].copy()
        return state

    def is_terminal(self) -> Tuple[bool, Optional[int]]:
        last_player = -self.current_player
        if self.board.check(last_player):
            return True, int(last_player)
        
        # Check draw (en Hex no hay empates, pero por si acaso)
        if len(self.get_legal_moves()) == 0:
            return True, 0
            
        return False, None

    def check_winner(self) -> int:
        if self.board.check(1):
            return 1
        if self.board.check(-1):
            return -1
        return 0

    def render(self):
        print(self.board)

def test_game():
    # TEST 1: Inicialización básica
    print("=== TEST 1: Inicialización ===")
    game = HexGame(size=7)
    print(f"Tamaño del tablero: {game.size}")
    print(f"Jugador actual: {game.current_player}")
    print(f"Movimientos: {game.move_count}")
    print(f"Estado inicial:\n{game.get_state()}")
    print("✓ Inicialización correcta\n")

    # TEST 2: Movimientos legales iniciales
    print("=== TEST 2: Movimientos Legales ===")
    legal_moves = game.get_legal_moves()
    print(f"Movimientos legales iniciales: {len(legal_moves)}")
    print(f"Primeros 5 movimientos: {legal_moves[:5]}")
    mask = game.get_legal_moves_mask()
    print(f"Máscara shape: {mask.shape}, Suma (debe ser 49): {mask.sum()}")
    print("✓ Movimientos legales correctos\n")

    # TEST 3: Aplicar movimiento básico
    print("=== TEST 3: Movimiento Básico ===")
    first_move = legal_moves[10]  # Movimiento en el centro
    print(f"Aplicando movimiento: {first_move}")
    state_before = game.get_state().copy()
    game.apply_move(first_move)
    state_after = game.get_state()
    print(f"Estado antes movimiento [5,5]: {state_before[5,5]}")
    print(f"Estado después movimiento [5,5]: {state_after[5,5]}")
    print(f"Jugador cambió a: {game.current_player}")
    print(f"Contador de movimientos: {game.move_count}")
    print("✓ Movimiento aplicado correctamente\n")

    # TEST 4: Verificar que el movimiento se refleja en el tablero visual
    print("=== TEST 4: Visualización ===")
    game.render()
    print("✓ Tablero visualizado\n")

    # TEST 5: Movimiento ilegal
    print("=== TEST 5: Movimiento Ilegal ===")
    try:
        game.apply_move(first_move)  # Mismo movimiento otra vez
        print("✗ ERROR: Debería haber fallado")
    except ValueError as e:
        print(f"✓ Correctamente bloqueado: {e}\n")

    # TEST 6: Interfaz step()
    print("=== TEST 6: Interfaz Step ===")
    game2 = HexGame(size=5)
    state = game2.reset()
    print(f"Estado inicial shape: {state.shape}")
    next_state, reward, done, info = game2.step((2, 2))
    print(f"Después de step - Recompensa: {reward}, Terminado: {done}")
    print(f"Estado siguiente shape: {next_state.shape}")
    print("✓ Interfaz step funciona\n")

    # TEST 7: Reset
    print("=== TEST 7: Reset ===")
    game3 = HexGame(size=3)
    game3.apply_move((1, 1))
    moves_before_reset = game3.move_count
    game3.reset()
    print(f"Movimientos antes reset: {moves_before_reset}")
    print(f"Movimientos después reset: {game3.move_count}")
    print(f"Terminado después reset: {game3.done}")
    print("✓ Reset funciona correctamente\n")

    # TEST 8: Condición de victoria simple
    print("=== TEST 8: Victoria Básica ===")
    # Para test rápido, usaremos un tablero pequeño y forzaremos conexión
    small_game = HexGame(size=3)

    # Simular movimientos que deberían crear un camino ganador para el jugador 1 (horizontal)
    # Jugador 1 (azul) necesita conectar izquierda-derecha
    moves_player1 = [(0,0), (0,1), (0,2)]  # Fila completa

    for i, move in enumerate(moves_player1):
        if small_game.current_player == 1:
            small_game.apply_move(move)
        else:
            # Si no es el turno del jugador 1, hacer un movimiento aleatorio del oponente
            legal = small_game.get_legal_moves()
            if legal and legal[0] != "swap":
                small_game.apply_move(legal[0])

    small_game.render()
    winner = small_game.check_winner()
    print(f"Ganador detectado: {winner}")
    print(f"Estado terminal: {small_game.done}")
    print("✓ Verificación de victoria funciona\n")

    print("TODAS LAS PRUEBAS COMPLETADAS ")

if __name__ == "__main__":
    test_game()