"""
Classe base abstrata para transmissores de modulação digital.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class BaseTransmitter(ABC):
    """
    Classe base abstrata para todos os transmissores.
    Define a interface comum para modulação digital.
    """
    
    def __init__(self, sample_rate: float, carrier_freq: float = 0):
        """
        Inicializa o transmissor.
        
        Args:
            sample_rate: Taxa de amostragem em Hz
            carrier_freq: Frequência da portadora em Hz (0 para banda base)
        """
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.samples_per_symbol = None
        
    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modula os bits de entrada em amostras I/Q complexas.
        
        Args:
            bits: Array de bits (0s e 1s)
            
        Returns:
            Array de amostras complexas (I + jQ)
        """
        pass
    
    def bits_from_bytes(self, data: bytes) -> np.ndarray:
        """
        Converte bytes em array de bits.
        
        Args:
            data: Dados em bytes
            
        Returns:
            Array numpy de bits (0s e 1s)
        """
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return bits
    
    def normalize_signal(self, signal: np.ndarray, max_amplitude: float = 0.9) -> np.ndarray:
        """
        Normaliza o sinal para evitar saturação.
        
        Args:
            signal: Sinal a ser normalizado
            max_amplitude: Amplitude máxima desejada (padrão 0.9 para margem)
            
        Returns:
            Sinal normalizado
        """
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal * (max_amplitude / max_val)
        return signal
    
    def upconvert(self, baseband_signal: np.ndarray) -> np.ndarray:
        """
        Converte sinal de banda base para frequência da portadora.
        
        Args:
            baseband_signal: Sinal em banda base (complexo)
            
        Returns:
            Sinal em RF (ainda complexo, para transmissão I/Q)
        """
        if self.carrier_freq == 0:
            return baseband_signal
            
        t = np.arange(len(baseband_signal)) / self.sample_rate
        carrier = np.exp(1j * 2 * np.pi * self.carrier_freq * t)
        return baseband_signal * carrier
    
    def get_modulation_type(self) -> str:
        """
        Retorna o tipo de modulação implementada.
        
        Returns:
            String com o nome da modulação (ex: 'BPSK', 'QPSK', etc.)
        """
        return self.__class__.__name__.replace('Transmitter', '').upper()
