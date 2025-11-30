"""
Classe base abstrata para canais de comunicação.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseChannel(ABC):
    """
    Classe base abstrata para canais de comunicação.
    Define a interface comum para simulação de canais.
    """
    
    def __init__(self):
        """
        Inicializa o canal.
        """
        self.parameters = {}
        
    @abstractmethod
    def transmit(self, signal: np.ndarray) -> np.ndarray:
        """
        Transmite o sinal através do canal.
        
        Args:
            signal: Sinal de entrada (complexo)
            
        Returns:
            Sinal de saída com efeitos do canal aplicados
        """
        pass
    
    @abstractmethod
    def get_channel_type(self) -> str:
        """
        Retorna o tipo de canal.
        
        Returns:
            String com o nome do canal
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros do canal.
        
        Returns:
            Dicionário com parâmetros do canal
        """
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs):
        """
        Atualiza parâmetros do canal.
        
        Args:
            **kwargs: Parâmetros a serem atualizados
        """
        self.parameters.update(kwargs)
