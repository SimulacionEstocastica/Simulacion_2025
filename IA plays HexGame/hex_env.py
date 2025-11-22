# hex_environment.py
import numpy as np
import torch
from typing import Tuple, List
from hex_game import HexGame

class HexEnvironment:
    def __init__(self, size: int = 7):
        self.size = size
        self.game = HexGame(size)
        self.action_size = size * size + 1  # +1 para swap
        
    def reset(self) -> torch.Tensor:
        """Reinicia el juego y retorna el estado como tensor"""
        self.game.reset()
        return self.state_to_tensor()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Ejecuta una acción y retorna (next_state, reward, done, info)
        action: índice de 0 a (size*size) para movimientos normales, size*size para swap
        """
        # Convertir acción índice a movimiento del juego
        move = self.action_to_move(action)
        
        # Ejecutar movimiento
        state, reward, done, info = self.game.step(move)
        next_state = self.state_to_tensor()
        
        return next_state, reward, done, info
    
    def action_to_move(self, action: int):
        """Convierte índice de acción a movimiento del juego"""
        if action == self.size * self.size:  # swap
            return "swap"
        else:
            # Convertir índice a coordenadas (fila, columna)
            row = action // self.size
            col = action % self.size
            return (row, col)
    
    def move_to_action(self, move) -> int:
        """Convierte movimiento del juego a índice de acción"""
        if move == "swap":
            return self.size * self.size
        else:
            row, col = move
            return row * self.size + col
    
    def state_to_tensor(self) -> torch.Tensor:
        """
        Convierte el estado del juego a tensor para la red neuronal
        Retorna tensor de shape [8, size, size] con 8 canales:
        Canal 0: Posiciones del jugador actual (1 donde hay piedras del jugador actual)
        Canal 1: Posiciones del oponente (1 donde hay piedras del oponente)
        Canal 2: Tablero vacío (1 donde no hay piedras)
        Canal 3: Turno actual (todo 1s si es jugador 1, todo 0s si es jugador -1)
        Canal 4: Movimientos legales (máscara)
        Canales 5-7: Canales de posición (coordenadas normalizadas)
        """
        board_state = self.game.get_state()
        current_player = self.game.current_player
        
        # Canal 0: Piedras del jugador actual
        if current_player == 1:
            player_channel = (board_state == 1).astype(np.float32)
            opponent_channel = (board_state == -1).astype(np.float32)
        else:
            player_channel = (board_state == -1).astype(np.float32)
            opponent_channel = (board_state == 1).astype(np.float32)
        
        # Canal 2: Casillas vacías
        empty_channel = (board_state == 0).astype(np.float32)
        
        # Canal 3: Turno (1 para jugador 1, 0 para jugador -1)
        turn_channel = np.full((self.size, self.size), 
                             1.0 if current_player == 1 else 0.0, 
                             dtype=np.float32)
        
        # Canal 4: Movimientos legales
        legal_moves = self.game.get_legal_moves()
        legal_channel = np.zeros((self.size, self.size), dtype=np.float32)
        for move in legal_moves:
            if move != "swap":
                legal_channel[move] = 1.0
        
        # Canales 5-7: Posiciones (coordenadas normalizadas)
        x_channel = np.zeros((self.size, self.size), dtype=np.float32)
        y_channel = np.zeros((self.size, self.size), dtype=np.float32)
        
        for i in range(self.size):
            for j in range(self.size):
                x_channel[i, j] = i / (self.size - 1) if self.size > 1 else 0.5
                y_channel[i, j] = j / (self.size - 1) if self.size > 1 else 0.5
        
        # Stack todos los canales
        state_tensor = np.stack([
            player_channel,      # Canal 0
            opponent_channel,    # Canal 1  
            empty_channel,       # Canal 2
            turn_channel,        # Canal 3
            legal_channel,       # Canal 4
            x_channel,           # Canal 5
            y_channel,           # Canal 6
        ], axis=0)
        
        return torch.from_numpy(state_tensor)
    
    def get_legal_actions_mask(self) -> torch.Tensor:
        """Retorna máscara de acciones legales como tensor"""
        mask = np.zeros(self.action_size, dtype=np.float32)
        legal_moves = self.game.get_legal_moves()
        
        for move in legal_moves:
            action_idx = self.move_to_action(move)
            mask[action_idx] = 1.0
            
        return torch.from_numpy(mask)
    
    def get_valid_action_indices(self) -> List[int]:
        """Retorna lista de índices de acciones válidas"""
        legal_moves = self.game.get_legal_moves()
        return [self.move_to_action(move) for move in legal_moves]
    
    def render(self):
        """Renderiza el tablero"""
        self.game.render()
    
    def is_terminal(self) -> bool:
        """Verifica si el juego terminó"""
        return self.game.done
    
    def get_winner(self) -> int:
        """Retorna el ganador (1, -1, o 0 si no hay)"""
        return self.game.winner

# Test del Environment
def test_environment():
    print("=== TEST Environment ===")
    
    # Test 1: Inicialización
    env = HexEnvironment(size=7)
    state = env.reset()
    print(f"Estado tensor shape: {state.shape}")  # Debería ser [8, 7, 7]
    print(f"Tamaño espacio de acciones: {env.action_size}")  # Debería ser 50
    
    # Test 2: Máscara de acciones legales
    legal_mask = env.get_legal_actions_mask()
    print(f"Máscara legal shape: {legal_mask.shape}")
    print(f"Acciones legales: {legal_mask.sum().item()}")
    
    # Test 3: Ejecutar acción
    action = 24  # Movimiento en el centro
    next_state, reward, done, info = env.step(action)
    print(f"Recompensa después movimiento: {reward}")
    print(f"Terminado: {done}")
    print(f"Nuevo estado shape: {next_state.shape}")
    
    # Test 4: Conversiones acción-movimiento
    test_action = 10
    move = env.action_to_move(test_action)
    recovered_action = env.move_to_action(move)
    print(f"Acción {test_action} -> Movimiento {move} -> Acción {recovered_action}")
    
    # Test 5: Swap
    env2 = HexEnvironment(size=3)
    env2.reset()
    env2.step(4)  # Primer movimiento
    print("Swap disponible después primer movimiento:", "swap" in env2.game.get_legal_moves())
    
    print("✓ Environment funcionando correctamente")

if __name__ == "__main__":
    test_environment()