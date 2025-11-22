# hex_mcts.py
# hex_mcts.py
import numpy as np
import torch
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import copy

class Node:
    """Nodo del árbol de MCTS"""
    def __init__(self, prior: float, parent: Optional['Node'] = None):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[int, 'Node'] = {}
        self.parent = parent
        self.state = None
        self.action = None
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, exploration_weight: float = 1.0) -> float:
        """Calcula el score UCB para este nodo"""
        if self.visit_count == 0:
            return float('inf')
        
        # Exploitation: valor promedio del nodo
        exploitation = self.value()
        
        # Exploration: prior * sqrt(parent_visits) / (1 + visit_count)
        exploration = self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration_weight * exploration

class MCTS:
    """Monte Carlo Tree Search con red neuronal para HEX"""
    def __init__(self, model, environment, num_simulations: int = 100,
                 exploration_weight: float = 1.0, device: str = 'cuda'):
        self.model = model
        self.env = environment
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.device = device
        
    def run(self, state: torch.Tensor, legal_actions_mask: torch.Tensor) -> np.ndarray:
        """
        Ejecuta MCTS desde el estado dado y retorna probabilidades de acción
        
        Args:
            state: Estado actual del juego (debe estar en el mismo dispositivo que el modelo)
            legal_actions_mask: Máscara de acciones legales
        """
        root = Node(0.0)
        
        # Asegurar que el estado esté en el dispositivo correcto
        state = state.to(self.device).unsqueeze(0)  # Añadir dimensión batch
        
        # Expandir el nodo raíz
        with torch.no_grad():
            policy_logits, value = self.model(state)
            policy_logits = policy_logits.squeeze(0).cpu().numpy()
        
        # Aplicar máscara de acciones legales y convertir a probabilidades
        legal_mask_np = legal_actions_mask.cpu().numpy()
        masked_policy = policy_logits + (1 - legal_mask_np) * -1e9
        policy_probs = np.exp(masked_policy) / np.sum(np.exp(masked_policy))
        
        # Crear nodos hijos para acciones legales
        for action in range(len(legal_mask_np)):
            if legal_mask_np[action] == 1:
                root.children[action] = Node(policy_probs[action], parent=root)
        
        # Realizar simulaciones
        for i in range(self.num_simulations):
            env_copy = copy.deepcopy(self.env)
            self._simulate(root, env_copy)
        
        # Calcular probabilidades basadas en conteos de visita
        visit_counts = np.zeros(self.env.action_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # Normalizar para obtener probabilidades
        if visit_counts.sum() > 0:
            action_probs = visit_counts / visit_counts.sum()
        else:
            action_probs = np.ones(self.env.action_size) / self.env.action_size
        
        return action_probs
    
    def _simulate(self, node: Node, env) -> float:
        """
        Realiza una simulación desde el nodo dado
        
        Returns:
            Valor estimado del estado final
        """
        # Selección: bajar por el árbol hasta encontrar un nodo no expandido
        while node.expanded() and not env.is_terminal():
            action, node = self._select_child(node)
            env.step(action)
        
        # Valor del estado terminal o evaluación con la red
        if env.is_terminal():
            winner = env.get_winner()
            # Convertir resultado a perspectiva del jugador actual
            current_player = env.game.current_player
            if winner == 0:  # Empate (no debería pasar en Hex)
                value = 0.0
            else:
                value = 1.0 if winner == current_player else -1.0
        else:
            # Evaluar con la red neuronal
            state_tensor = env.state_to_tensor().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                value = value.item()
            
            # Expandir el nodo actual
            legal_actions_mask = env.get_legal_actions_mask()
            policy_logits = policy_logits.squeeze(0).cpu().numpy()
            legal_mask_np = legal_actions_mask.cpu().numpy()
            
            # Aplicar máscara y crear nodos hijos
            masked_policy = policy_logits + (1 - legal_mask_np) * -1e9
            policy_probs = np.exp(masked_policy) / np.sum(np.exp(masked_policy))
            
            for action in range(len(legal_mask_np)):
                if legal_mask_np[action] == 1:
                    node.children[action] = Node(policy_probs[action], parent=node)
        
        # Retropropagación
        self._backpropagate(node, value)
        return value
    
    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """Selecciona el hijo con mayor score UCB"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            score = child.ucb_score(self.exploration_weight)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _backpropagate(self, node: Node, value: float) -> None:
        """Retropropaga el valor a través del árbol"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value  # Alternar perspectiva entre jugadores
            current = current.parent

# Función para seleccionar acción durante el juego
def select_action(action_probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Selecciona una acción basada en las probabilidades y temperatura
    
    Args:
        action_probs: Probabilidades de cada acción
        temperature: Controla la exploración (1.0 = proporcional, 0.0 = greedy)
    
    Returns:
        Índice de la acción seleccionada
    """
    if temperature == 0.0:
        # Selección greedy
        return np.argmax(action_probs)
    else:
        # Aplicar temperatura y muestrear
        probs = action_probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

# Test del MCTS corregido
def test_mcts():
    print("=== TEST MCTS ===")
    
    from hex_network import create_hex_network
    from hex_env import HexEnvironment
    
    # Configuración
    board_size = 7
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Crear modelo y environment
    model = create_hex_network(device=device)
    env = HexEnvironment(board_size)
    
    # Crear MCTS (pocas simulaciones para test rápido)
    mcts = MCTS(model, env, num_simulations=10, device=device)
    
    # Estado inicial - asegurar que esté en el dispositivo correcto
    state = env.reset()
    legal_actions_mask = env.get_legal_actions_mask()
    
    print(f"Estado shape: {state.shape}")
    print(f"Acciones legales: {legal_actions_mask.sum().item()}")
    print(f"Dispositivo del modelo: {next(model.parameters()).device}")
    
    # Ejecutar MCTS
    action_probs = mcts.run(state, legal_actions_mask)
    
    print(f"Probabilidades de acción shape: {action_probs.shape}")
    print(f"Suma de probabilidades: {action_probs.sum():.6f}")
    print(f"Acción con mayor probabilidad: {np.argmax(action_probs)}")
    print(f"Valor máximo de probabilidad: {action_probs.max():.6f}")
    
    # Probar selección de acción
    action_greedy = select_action(action_probs, temperature=0.0)
    action_exploratory = select_action(action_probs, temperature=1.0)
    
    print(f"Acción greedy: {action_greedy}")
    print(f"Acción exploratoria: {action_exploratory}")
    
    # Verificar propiedades
    assert abs(action_probs.sum() - 1.0) < 1e-6, "Las probabilidades deben sumar 1"
    assert (action_probs >= 0).all(), "Las probabilidades deben ser no negativas"
    
    print("✓ MCTS funcionando correctamente")

if __name__ == "__main__":
    test_mcts()