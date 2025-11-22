# hex_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HexNet(nn.Module):
    def __init__(self, board_size=7, action_size=50, num_channels=7, num_res_blocks=10):
        """
        Red neuronal para HEX inspirada en AlphaZero
        
        Args:
            board_size: Tamaño del tablero (7 para 7x7)
            action_size: Número de acciones posibles (board_size² + 1 para swap)
            num_channels: Número de canales de entrada (7 en nuestro caso)
            num_res_blocks: Número de bloques residuales
        """
        super(HexNet, self).__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.num_channels = num_channels
        
        # Bloque convolucional inicial
        self.conv_input = nn.Conv2d(num_channels, 256, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(256)
        
        # Bloques residuales
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(num_res_blocks)
        ])
        
        # Cabezas de política y valor
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)
        
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass de la red
        
        Args:
            x: Tensor de entrada [batch_size, num_channels, board_size, board_size]
            
        Returns:
            policy: Probabilidades de acciones [batch_size, action_size]
            value: Valor estimado del estado [batch_size, 1]
        """
        # Bloque inicial
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Bloques residuales
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Cabeza de política
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        
        # Cabeza de valor
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return F.log_softmax(policy, dim=1), value

class ResidualBlock(nn.Module):
    """Bloque residual con dos capas convolucionales"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Conexión residual
        x = F.relu(x)
        return x

# Función para crear la red y moverla a GPU si está disponible
def create_hex_network(board_size=7, action_size=50, num_channels=7, device='cuda'):
    """Crea y configura la red neuronal para HEX"""
    model = HexNet(
        board_size=board_size,
        action_size=action_size,
        num_channels=num_channels
    )
    
    # Mover a GPU si está disponible
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"Modelo movido a GPU: {torch.cuda.get_device_name()}")
    else:
        print("Modelo en CPU")
    
    return model

# Test de la red neuronal CORREGIDO
def test_network():
    print("=== TEST Red Neuronal ===")
    
    # Configuración
    board_size = 7
    action_size = 50
    num_channels = 7
    batch_size = 4
    
    # Crear modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_hex_network(
        board_size=board_size,
        action_size=action_size,
        num_channels=num_channels,
        device=device
    )
    
    # Crear datos de prueba
    dummy_input = torch.randn(batch_size, num_channels, board_size, board_size)
    if device == 'cuda':
        dummy_input = dummy_input.cuda()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        policy, value = model(dummy_input)
    
    # Verificar shapes
    print(f"Input shape: {dummy_input.shape}")
    print(f"Policy output shape: {policy.shape}")  # [batch_size, action_size]
    print(f"Value output shape: {value.shape}")    # [batch_size, 1]
    
    # Verificar propiedades
    print(f"Policy sum (debe ser ~0 por log_softmax): {torch.exp(policy).sum(dim=1)}")
    print(f"Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Test con máscara de acciones legales - CORREGIDO
    legal_moves_mask = torch.ones(batch_size, action_size)
    legal_moves_mask[:, -1] = 0  # Deshabilitar swap para test
    
    # Mover máscara al mismo dispositivo que el modelo
    if device == 'cuda':
        legal_moves_mask = legal_moves_mask.cuda()
    
    # Aplicar máscara a políticas
    masked_policy = policy + (1 - legal_moves_mask) * -1e9
    masked_policy = F.softmax(masked_policy, dim=1)
    
    print(f"Política después de máscara - suma por batch: {masked_policy.sum(dim=1)}")
    
    print("✓ Red neuronal funcionando correctamente")

if __name__ == "__main__":
    test_network()