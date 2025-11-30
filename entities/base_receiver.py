"""
Classe base abstrata para receptores de modulação digital.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseReceiver(ABC):
    """
    Classe base abstrata para todos os receptores.
    Define a interface comum para demodulação digital.
    """
    
    def __init__(self, sample_rate: float, carrier_freq: float = 0):
        """
        Inicializa o receptor.
        
        Args:
            sample_rate: Taxa de amostragem em Hz
            carrier_freq: Frequência da portadora em Hz (0 para banda base)
        """
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.samples_per_symbol = None
        
    @abstractmethod
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """
        Demodula o sinal recebido em bits.
        
        Args:
            signal: Sinal I/Q recebido (complexo)
            
        Returns:
            Array de bits recuperados (0s e 1s)
        """
        pass
    
    def downconvert(self, rf_signal: np.ndarray) -> np.ndarray:
        """
        Converte sinal de RF para banda base.
        
        Args:
            rf_signal: Sinal em RF (complexo)
            
        Returns:
            Sinal em banda base (complexo)
        """
        if self.carrier_freq == 0:
            return rf_signal
            
        t = np.arange(len(rf_signal)) / self.sample_rate
        carrier = np.exp(-1j * 2 * np.pi * self.carrier_freq * t)
        return rf_signal * carrier
    
    def bytes_from_bits(self, bits: np.ndarray) -> bytes:
        """
        Converte array de bits em bytes.
        
        Args:
            bits: Array de bits (0s e 1s)
            
        Returns:
            Dados em bytes
        """
        # Garante que o número de bits é múltiplo de 8
        if len(bits) % 8 != 0:
            padding = 8 - (len(bits) % 8)
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        # Converte para bytes
        bits_uint8 = bits.astype(np.uint8)
        byte_array = np.packbits(bits_uint8)
        return byte_array.tobytes()
    
    def decision_directed_sync(self, signal: np.ndarray, pilot_length: int = 100) -> Tuple[float, float]:
        """
        Estimação de fase e frequência usando símbolos piloto.
        
        Args:
            signal: Sinal recebido
            pilot_length: Comprimento da sequência piloto
            
        Returns:
            Tupla (phase_offset, freq_offset)
        """
        # Implementação simplificada - pode ser sobrescrita por subclasses
        phase_offset = 0.0
        freq_offset = 0.0
        
        return phase_offset, freq_offset
    
    def timing_recovery(self, signal: np.ndarray) -> np.ndarray:
        """
        Recuperação de temporização (clock recovery).
        
        Args:
            signal: Sinal recebido
            
        Returns:
            Sinal com símbolos alinhados temporalmente
        """
        # Implementação simplificada - Mueller-Muller ou Gardner pode ser usado
        # Por enquanto, apenas retorna o sinal
        return signal
    
    def apply_agc(self, signal: np.ndarray, target_power: float = 1.0) -> np.ndarray:
        """
        Controle Automático de Ganho (AGC).
        
        Args:
            signal: Sinal recebido
            target_power: Potência alvo desejada
            
        Returns:
            Sinal com ganho ajustado
        """
        current_power = np.mean(np.abs(signal)**2)
        
        if current_power > 0:
            gain = np.sqrt(target_power / current_power)
            return signal * gain
        
        return signal
    
    def get_modulation_type(self) -> str:
        """
        Retorna o tipo de modulação implementada.
        
        Returns:
            String com o nome da modulação
        """
        return self.__class__.__name__.replace('Receiver', '').upper()
