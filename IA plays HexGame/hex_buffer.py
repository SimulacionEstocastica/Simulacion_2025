# hex_buffer.py
import numpy as np
import torch
from collections import deque
import random
from typing import List, Tuple, Deque, Any

class ExperienceBuffer:
    """
    Buffer de experiencia para almacenar partidas completas de HEX
    Almacena: (estado, políticas MCTS, resultado)
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        """Añade una experiencia al buffer"""
        self.buffer.append((state, policy, value))
    
    def add_game(self, game_history: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Añade una partida completa al buffer"""
        for state, policy, value in game_history:
            self.add(state, policy, value)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Muestrea un batch aleatorio del buffer"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        
        # Convertir a tensores de PyTorch
        states_tensor = torch.FloatTensor(np.array(states))
        policies_tensor = torch.FloatTensor(np.array(policies))
        values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1)
        
        return states_tensor, policies_tensor, values_tensor
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Limpia el buffer"""
        self.buffer.clear()

# Clase para gestionar partidas completas
class GameHistory:
    """Almacena el historial completo de una partida"""
    
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.mcts_policies: List[np.ndarray] = []
        self.current_player: List[int] = []
    
    def add_step(self, state: np.ndarray, policy: np.ndarray, player: int):
        """Añade un paso del juego al historial"""
        self.states.append(state.copy())
        self.mcts_policies.append(policy.copy())
        self.current_player.append(player)
    
    def get_training_data(self, final_result: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Convierte el historial en datos de entrenamiento
        final_result: resultado desde la perspectiva del jugador en cada estado
        """
        training_data = []
        for i in range(len(self.states)):
            # El resultado desde la perspectiva del jugador en ese momento
            if self.current_player[i] == 1:
                value = final_result
            else:
                value = -final_result  # Invertir perspectiva
            
            training_data.append((self.states[i], self.mcts_policies[i], value))
        
        return training_data

# Test del Buffer de Experiencia
def test_buffer():
    print("=== TEST Buffer de Experiencia ===")
    
    buffer = ExperienceBuffer(capacity=100)
    
    # Crear datos de prueba
    state_shape = (7, 7, 7)
    action_size = 50
    
    # Añadir algunas experiencias
    for i in range(5):
        state = np.random.randn(*state_shape).astype(np.float32)
        policy = np.random.randn(action_size).astype(np.float32)
        policy = np.exp(policy) / np.sum(np.exp(policy))  # Convertir a probabilidades
        value = np.random.uniform(-1, 1)
        
        buffer.add(state, policy, value)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Samplear un batch
    batch_size = 3
    states, policies, values = buffer.sample(batch_size)
    
    print(f"Batch states shape: {states.shape}")
    print(f"Batch policies shape: {policies.shape}")
    print(f"Batch values shape: {values.shape}")
    
    # Verificar propiedades
    assert states.shape == (batch_size, *state_shape)
    assert policies.shape == (batch_size, action_size)
    assert values.shape == (batch_size, 1)
    
    # Verificar que las políticas suman 1
    policy_sums = policies.sum(dim=1)
    print(f"Sumas de políticas: {policy_sums}")
    assert torch.allclose(policy_sums, torch.ones_like(policy_sums), atol=1e-5)
    
    # Test GameHistory
    print("\n--- Test GameHistory ---")
    game_history = GameHistory()
    
    # Simular una partida corta
    for i in range(3):
        state = np.random.randn(*state_shape).astype(np.float32)
        policy = np.random.randn(action_size).astype(np.float32)
        policy = np.exp(policy) / np.sum(np.exp(policy))
        player = 1 if i % 2 == 0 else -1
        
        game_history.add_step(state, policy, player)
    
    # Obtener datos de entrenamiento (suponiendo que ganó el jugador 1)
    training_data = game_history.get_training_data(final_result=1.0)
    print(f"Training data length: {len(training_data)}")
    
    # Añadir la partida completa al buffer
    buffer.add_game(training_data)
    print(f"Buffer size después de añadir partida: {len(buffer)}")
    
    # Verificar que los valores tienen la perspectiva correcta
    for i, (state, policy, value) in enumerate(training_data):
        expected_value = 1.0 if game_history.current_player[i] == 1 else -1.0
        print(f"Paso {i}: jugador {game_history.current_player[i]}, valor {value}, esperado {expected_value}")
        assert abs(value - expected_value) < 1e-6
    
    print("✓ Buffer de experiencia funcionando correctamente")

if __name__ == "__main__":
    test_buffer()