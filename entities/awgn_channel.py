"""
Canal AWGN (Additive White Gaussian Noise).
Adiciona ruído branco Gaussiano ao sinal transmitido.
"""
import numpy as np
from .base_channel import BaseChannel
from typing import Optional


class AWGNChannel(BaseChannel):
    """
    Canal AWGN (Additive White Gaussian Noise).
    
    Modelo de canal mais simples, adiciona apenas ruído branco Gaussiano.
    Usado como referência em análises teóricas de desempenho.
    
    SNR (Signal-to-Noise Ratio) pode ser especificado em dB ou linear.
    """
    
    def __init__(self, snr_db: float = 10.0, noise_power: Optional[float] = None):
        """
        Inicializa o canal AWGN.
        
        Args:
            snr_db: Relação Sinal-Ruído em dB (padrão: 10 dB)
            noise_power: Potência do ruído (linear). Se especificado, ignora snr_db.
        """
        super().__init__()
        
        self.parameters = {
            'snr_db': snr_db,
            'noise_power': noise_power,
            'noise_samples_generated': 0
        }
        
    def transmit(self, signal: np.ndarray) -> np.ndarray:
        """
        Transmite o sinal através do canal AWGN.
        
        Args:
            signal: Sinal complexo de entrada (I + jQ)
            
        Returns:
            Sinal com ruído AWGN adicionado
        """
        # Calcula potência do sinal
        signal_power = np.mean(np.abs(signal)**2)
        
        # Determina potência do ruído
        if self.parameters['noise_power'] is not None:
            noise_power = self.parameters['noise_power']
        else:
            # Converte SNR de dB para linear
            snr_linear = 10**(self.parameters['snr_db'] / 10.0)
            noise_power = signal_power / snr_linear
        
        # Gera ruído branco Gaussiano complexo
        noise = self._generate_awgn(len(signal), noise_power)
        
        # Adiciona ruído ao sinal
        noisy_signal = signal + noise
        
        # Atualiza estatísticas
        self.parameters['noise_samples_generated'] += len(signal)
        
        return noisy_signal
    
    def _generate_awgn(self, num_samples: int, noise_power: float) -> np.ndarray:
        """
        Gera ruído AWGN complexo.
        
        Args:
            num_samples: Número de amostras
            noise_power: Potência do ruído
            
        Returns:
            Ruído complexo (I + jQ)
        """
        # Ruído Gaussiano com variância = noise_power/2 em cada componente (I e Q)
        # Para que a potência total seja noise_power
        std_dev = np.sqrt(noise_power / 2.0)
        
        noise_i = np.random.normal(0, std_dev, num_samples)
        noise_q = np.random.normal(0, std_dev, num_samples)
        
        noise = noise_i + 1j * noise_q
        
        return noise.astype(np.complex64)
    
    def set_snr_db(self, snr_db: float):
        """
        Atualiza a SNR do canal.
        
        Args:
            snr_db: Nova SNR em dB
        """
        self.parameters['snr_db'] = snr_db
        self.parameters['noise_power'] = None  # Força recálculo
    
    def set_noise_power(self, noise_power: float):
        """
        Define diretamente a potência do ruído.
        
        Args:
            noise_power: Potência do ruído (linear)
        """
        self.parameters['noise_power'] = noise_power
    
    def get_snr_db(self) -> float:
        """
        Retorna a SNR configurada em dB.
        
        Returns:
            SNR em dB
        """
        return self.parameters['snr_db']
    
    def get_theoretical_ber(self, modulation: str = 'BPSK') -> float:
        """
        Calcula a BER teórica para o canal AWGN.
        
        Args:
            modulation: Tipo de modulação ('BPSK', '16QAM', etc.)
            
        Returns:
            BER teórica
        """
        from scipy.special import erfc
        
        snr_linear = 10**(self.parameters['snr_db'] / 10.0)
        
        if modulation.upper() == 'BPSK':
            # BER para BPSK: Q(sqrt(2*Eb/N0))
            ber = 0.5 * erfc(np.sqrt(snr_linear))
            
        elif modulation.upper() in ['16QAM', 'QAM16']:
            # BER aproximada para 16-QAM (Gray coding)
            # BER ≈ (3/8) * erfc(sqrt(0.8 * SNR))
            ber = (3.0/8.0) * erfc(np.sqrt(0.8 * snr_linear))
            
        elif modulation.upper() == 'QPSK':
            # BER para QPSK (equivalente a BPSK por bit)
            ber = 0.5 * erfc(np.sqrt(snr_linear))
            
        else:
            raise ValueError(f"Modulação '{modulation}' não suportada para cálculo teórico")
        
        return ber
    
    def measure_snr(self, signal: np.ndarray, noisy_signal: np.ndarray) -> float:
        """
        Mede a SNR real entre sinal original e sinal com ruído.
        
        Args:
            signal: Sinal original
            noisy_signal: Sinal com ruído
            
        Returns:
            SNR medida em dB
        """
        signal_power = np.mean(np.abs(signal)**2)
        noise = noisy_signal - signal
        noise_power = np.mean(np.abs(noise)**2)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = float('inf')
        
        return snr_db
    
    def get_channel_type(self) -> str:
        """
        Retorna o tipo de canal.
        
        Returns:
            String 'AWGN'
        """
        return 'AWGN'
    
    def reset_statistics(self):
        """
        Reseta estatísticas do canal.
        """
        self.parameters['noise_samples_generated'] = 0
    
    def __str__(self) -> str:
        return (f"AWGN Channel:\n"
                f"  SNR: {self.parameters['snr_db']:.2f} dB\n"
                f"  Noise Power: {self.parameters['noise_power']}\n"
                f"  Samples Generated: {self.parameters['noise_samples_generated']}")
