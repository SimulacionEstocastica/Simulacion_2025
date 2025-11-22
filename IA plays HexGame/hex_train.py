# hex_train_simple.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from hex_network import create_hex_network
from hex_env import HexEnvironment
from hex_mcts import MCTS, select_action
from hex_buffer import ExperienceBuffer, GameHistory

class SimpleHexTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = config['device']
        
        # Setup logging
        self.setup_logging()
        
        # Inicializar componentes
        self.model = create_hex_network(
            board_size=config['board_size'],
            action_size=config['action_size'],
            num_channels=config['num_channels'],
            device=config['device']
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
        self.buffer = ExperienceBuffer(capacity=config['buffer_capacity'])
        
        # Crear directorio para checkpoints
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Métricas
        self.training_step = 0
        self.games_played = 0
        self.metrics = {
            'policy_loss': [],
            'value_loss': [], 
            'total_loss': [],
            'game_lengths': [],
            'player1_wins': [],
            'player2_wins': [],
            'timestamps': []
        }
        
        self.logger.info("Entrenador inicializado correctamente")
        
    def setup_logging(self):
        """Configura logging simple"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def self_play_game(self) -> GameHistory:
        """Juega una partida de auto-jugada"""
        env = HexEnvironment(self.config['board_size'])
        mcts = MCTS(
            model=self.model,
            environment=env,
            num_simulations=self.config['num_simulations'],
            device=self.device
        )
        
        game_history = GameHistory()
        state = env.reset()
        
        while not env.is_terminal():
            legal_actions_mask = env.get_legal_actions_mask()
            action_probs = mcts.run(state, legal_actions_mask)
            
            game_history.add_step(
                state=state.cpu().numpy(),
                policy=action_probs,
                player=env.game.current_player
            )
            
            # Temperatura decreciente
            if len(game_history.states) < self.config['temperature_threshold']:
                temperature = 1.0
            else:
                temperature = 0.1
                
            action = select_action(action_probs, temperature)
            state, reward, done, info = env.step(action)
        
        winner = env.get_winner()
        final_result = 1.0 if winner == 1 else -1.0
        
        return game_history, final_result
    
    def train_batch(self, states: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> Tuple[float, float]:
        """Entrena con un batch de datos"""
        self.model.train()
        
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        policy_logits, values = self.model(states)
        
        policy_loss = -torch.sum(target_policies * policy_logits) / target_policies.size(0)
        value_loss = torch.nn.functional.mse_loss(values, target_values)
        total_loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def save_checkpoint(self, epoch: int, milestone: str = ""):
        """Guarda checkpoint del modelo y métricas"""
        if milestone:
            filename = f"hex_model_epoch_{epoch+1}_{milestone}.pth"
        else:
            filename = f"hex_model_epoch_{epoch+1}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'games_played': self.games_played,
            'metrics': self.metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar métricas en JSON
        metrics_path = os.path.join(self.checkpoint_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Checkpoint guardado: {checkpoint_path}")
        self.logger.info(f"Metricas guardadas: {metrics_path}")
    
    def train(self):
        """Loop principal de entrenamiento"""
        self.logger.info("Iniciando entrenamiento...")
        start_time = time.time()
        
        total_games = self.config['num_epochs'] * self.config['games_per_epoch']
        milestones = {
            '25pct': total_games // 4,
            '50pct': total_games // 2, 
            '75pct': 3 * total_games // 4,
            '100pct': total_games
        }
        
        self.logger.info(f"Milestones: {milestones}")
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            # Fase de auto-jugada
            self.logger.info(f"Epoca {epoch + 1}/{self.config['num_epochs']} - Auto-jugada")
            
            game_histories = []
            for i in range(self.config['games_per_epoch']):
                game_history, final_result = self.self_play_game()
                training_data = game_history.get_training_data(final_result)
                self.buffer.add_game(training_data)
                game_histories.append((len(game_history.states), final_result))
                self.games_played += 1
                
                # Progress cada 10 partidas
                if (i + 1) % 10 == 0:
                    self.logger.info(f"  Partidas jugadas: {i + 1}/{self.config['games_per_epoch']}")
            
            # Estadísticas
            game_lengths = [length for length, _ in game_histories]
            results = [result for _, result in game_histories]
            avg_game_length = np.mean(game_lengths)
            player1_wins = results.count(1.0)
            player2_wins = results.count(-1.0)
            
            self.metrics['game_lengths'].append(avg_game_length)
            self.metrics['player1_wins'].append(player1_wins)
            self.metrics['player2_wins'].append(player2_wins)
            self.metrics['timestamps'].append(time.time())
            
            self.logger.info(f"  Longitud promedio: {avg_game_length:.1f}")
            self.logger.info(f"  Victorias - Jugador 1: {player1_wins}, Jugador 2: {player2_wins}")
            
            # Fase de entrenamiento
            self.logger.info("  Entrenamiento...")
            total_policy_loss = 0
            total_value_loss = 0
            
            for i in range(self.config['training_steps_per_epoch']):
                states, policies, values = self.buffer.sample(self.config['batch_size'])
                policy_loss, value_loss = self.train_batch(states, policies, values)
                
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                self.training_step += 1
                
                # Progress cada 20 pasos
                if (i + 1) % 20 == 0:
                    self.logger.info(f"    Paso {i + 1}/{self.config['training_steps_per_epoch']}")
            
            # Pérdidas
            avg_policy_loss = total_policy_loss / self.config['training_steps_per_epoch']
            avg_value_loss = total_value_loss / self.config['training_steps_per_epoch']
            total_loss = avg_policy_loss + avg_value_loss
            
            self.metrics['policy_loss'].append(avg_policy_loss)
            self.metrics['value_loss'].append(avg_value_loss)
            self.metrics['total_loss'].append(total_loss)
            
            self.logger.info(f"  Perdidas - Politica: {avg_policy_loss:.4f}, Valor: {avg_value_loss:.4f}, Total: {total_loss:.4f}")
            
            # Guardar checkpoint en hitos
            current_total_games = self.games_played
            for milestone_name, milestone_games in milestones.items():
                if current_total_games >= milestone_games:
                    # Verificar que no hayamos guardado ya este hito
                    milestone_path = os.path.join(self.checkpoint_dir, f"hex_model_epoch_{epoch+1}_{milestone_name}.pth")
                    if not os.path.exists(milestone_path):
                        self.save_checkpoint(epoch, milestone_name)
                        self.logger.info(f"  HITO ALCANZADO: {milestone_name} ({milestone_games} partidas)")
                        # Remover del diccionario para no guardar de nuevo
                        milestones.pop(milestone_name)
                    break
            
            # Guardar checkpoint regular
            if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch)
            
            epoch_time = time.time() - epoch_start
            self.logger.info(f"  Epoca completada en {epoch_time:.1f}s")
            
            # Log cada 5 épocas
            if (epoch + 1) % 5 == 0:
                avg_loss = np.mean(self.metrics['total_loss'][-5:])
                avg_game_len = np.mean(self.metrics['game_lengths'][-5:])
                self.logger.info(f"RESUMEN 5 EPOCAS - Perdida promedio: {avg_loss:.4f}, Longitud promedio: {avg_game_len:.1f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"ENTRENAMIENTO COMPLETADO")
        self.logger.info(f"Tiempo total: {total_time:.1f}s")
        self.logger.info(f"Partidas jugadas: {self.games_played}")
        self.logger.info(f"Pasos de entrenamiento: {self.training_step}")
        
        # Guardar modelo final
        self.save_checkpoint(self.config['num_epochs'] - 1, "final")
        
        return self.metrics

# Configuración simple
def get_simple_config():
    return {
        'board_size': 7,
        'action_size': 50,
        'num_channels': 7,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Auto-jugada
        'num_simulations': 100,
        'games_per_epoch': 50,
        'temperature_threshold': 10,
        
        # Entrenamiento
        'num_epochs': 100,
        'training_steps_per_epoch': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'buffer_capacity': 10000,
        
        # Checkpoints
        'checkpoint_frequency': 10,
        'checkpoint_dir': 'checkpoints'
    }

# Test simple
def test_simple_trainer():
    print("=== TEST Entrenador Simple ===")
    
    config = get_simple_config()
    config.update({
        'games_per_epoch': 2,
        'num_epochs': 2,
        'training_steps_per_epoch': 2,
        'batch_size': 2,
        'num_simulations': 10
    })
    
    trainer = SimpleHexTrainer(config)
    
    print("Test auto-jugada...")
    game_history, final_result = trainer.self_play_game()
    print(f"Partida terminada. Longitud: {len(game_history.states)}, Resultado: {final_result}")
    
    print("Test entrenamiento...")
    states = torch.randn(2, 7, 7, 7)
    policies = torch.randn(2, 50)
    policies = torch.softmax(policies, dim=1)
    values = torch.randn(2, 1)
    
    policy_loss, value_loss = trainer.train_batch(states, policies, values)
    print(f"Perdidas - Politica: {policy_loss:.4f}, Valor: {value_loss:.4f}")
    
    print("Test sistema de guardado...")
    trainer.save_checkpoint(0, "test")
    
    print("✓ Entrenador simple funcionando correctamente")

if __name__ == "__main__":
    # test_simple_trainer()
    
    # Para entrenamiento completo, descomenta:
    config = get_simple_config()
    trainer = SimpleHexTrainer(config)
    metrics = trainer.train()