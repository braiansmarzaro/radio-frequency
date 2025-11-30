"""
Receptor 16-QAM (16-Quadrature Amplitude Modulation).
Demodula sinais 16-QAM recuperando 4 bits por símbolo.
"""
import numpy as np
from .base_receiver import BaseReceiver
from typing import Optional


class QAM16Receiver(BaseReceiver):
    """
    Receptor 16-QAM (16-Quadrature Amplitude Modulation).
    
    Demodula sinais 16-QAM:
    - Recupera 4 bits por símbolo
    - Usa decisão de mínima distância (hard decision)
    - Suporta equalização opcional
    
    Taxa de recepção: 4 bits por símbolo
    """
    
    def __init__(self, sample_rate: float, symbol_rate: float, carrier_freq: float = 0):
        """
        Inicializa o receptor 16-QAM.
        
        Args:
            sample_rate: Taxa de amostragem em Hz
            symbol_rate: Taxa de símbolos em Hz
            carrier_freq: Frequência da portadora em Hz (0 para banda base)
        """
        super().__init__(sample_rate, carrier_freq)
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        
        # Gera constelação de referência
        self.constellation = self._generate_constellation()
        
    def _generate_constellation(self) -> np.ndarray:
        """
        Gera a constelação 16-QAM de referência (mesma do transmissor).
        
        Returns:
            Array com 16 símbolos complexos
        """
        levels = np.array([-3, -1, 1, 3])
        
        constellation = np.zeros(16, dtype=np.complex64)
        
        # Mapeamento Gray-coded (mesmo do transmissor)
        gray_map = [
            0b0000, 0b0001, 0b0011, 0b0010,
            0b0100, 0b0101, 0b0111, 0b0110,
            0b1100, 0b1101, 0b1111, 0b1110,
            0b1000, 0b1001, 0b1011, 0b1010
        ]
        
        idx = 0
        for i in range(4):
            for q in range(4):
                gray_idx = gray_map[idx]
                constellation[gray_idx] = levels[q] + 1j * levels[3-i]
                idx += 1
        
        # Normaliza (mesma normalização do transmissor)
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        
        return constellation
    
    def demodulate(self, signal: np.ndarray, use_rrc: bool = False) -> np.ndarray:
        """
        Demodula sinal 16-QAM.
        
        Args:
            signal: Sinal I/Q recebido (complexo)
            use_rrc: Se True, aplica filtro casado RRC
            
        Returns:
            Array de bits recuperados
        """
        # 1. Downconvert (se necessário)
        baseband_signal = self.downconvert(signal)
        
        # 2. AGC (Controle Automático de Ganho)
        baseband_signal = self.apply_agc(baseband_signal)
        
        # 3. Filtro casado (se RRC foi usado na transmissão)
        if use_rrc:
            baseband_signal = self._apply_matched_filter(baseband_signal)
        
        # 4. Amostragem nos instantes corretos
        symbols = self._sample_symbols(baseband_signal)
        
        # 5. Decisão: mapeia símbolos recebidos para bits
        bits = self._symbol_to_bits(symbols)
        
        return bits
    
    def _apply_matched_filter(self, signal: np.ndarray, alpha: float = 0.35, span: int = 10) -> np.ndarray:
        """
        Aplica filtro casado RRC (mesmo do transmissor).
        
        Args:
            signal: Sinal a ser filtrado
            alpha: Fator de roll-off
            span: Número de símbolos
            
        Returns:
            Sinal filtrado
        """
        # Gera filtro RRC
        rrc_filter = self._rrc_coefficients(alpha, span)
        
        # Filtragem separada para I e Q
        signal_i = np.convolve(np.real(signal), rrc_filter, mode='same')
        signal_q = np.convolve(np.imag(signal), rrc_filter, mode='same')
        
        return signal_i + 1j * signal_q
    
    def _rrc_coefficients(self, alpha: float, span: int) -> np.ndarray:
        """
        Calcula coeficientes do filtro Root Raised Cosine.
        
        Args:
            alpha: Fator de roll-off
            span: Número de símbolos
            
        Returns:
            Coeficientes do filtro
        """
        N = span * self.samples_per_symbol
        t = np.arange(-N//2, N//2) / self.samples_per_symbol
        
        h = np.zeros(len(t))
        
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = (1 + alpha * (4/np.pi - 1))
            elif abs(abs(ti) - 1/(4*alpha)) < 1e-10:
                h[i] = (alpha/np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
                )
            else:
                numerator = np.sin(np.pi * ti * (1 - alpha)) + \
                           4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
                denominator = np.pi * ti * (1 - (4 * alpha * ti)**2)
                h[i] = numerator / denominator
        
        # Normaliza
        h = h / np.sqrt(np.sum(h**2))
        
        return h
    
    def _sample_symbols(self, signal: np.ndarray) -> np.ndarray:
        """
        Amostra o sinal nos instantes de símbolo.
        
        Args:
            signal: Sinal em banda base
            
        Returns:
            Array de símbolos complexos
        """
        # Amostragem simples: pega 1 amostra a cada samples_per_symbol
        # Em um receptor real, seria necessário timing recovery
        num_symbols = len(signal) // self.samples_per_symbol
        symbols = signal[::self.samples_per_symbol][:num_symbols]
        
        return symbols
    
    def _symbol_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converte símbolos recebidos em bits usando decisão de mínima distância.
        
        Args:
            symbols: Símbolos complexos recebidos
            
        Returns:
            Array de bits
        """
        num_symbols = len(symbols)
        bits = np.zeros(num_symbols * 4, dtype=int)
        
        for i, symbol in enumerate(symbols):
            # Decisão de mínima distância (hard decision)
            distances = np.abs(symbol - self.constellation)
            min_idx = np.argmin(distances)
            
            # Converte índice para 4 bits
            bits[i*4] = (min_idx >> 3) & 1
            bits[i*4 + 1] = (min_idx >> 2) & 1
            bits[i*4 + 2] = (min_idx >> 1) & 1
            bits[i*4 + 3] = min_idx & 1
        
        return bits
    
    def soft_decision_demodulate(self, signal: np.ndarray, use_rrc: bool = False) -> np.ndarray:
        """
        Demodulação com decisão suave (soft decision) - retorna LLRs.
        
        Args:
            signal: Sinal I/Q recebido
            use_rrc: Se True, aplica filtro casado RRC
            
        Returns:
            Log-Likelihood Ratios (LLRs) para cada bit
        """
        # Processamento inicial
        baseband_signal = self.downconvert(signal)
        baseband_signal = self.apply_agc(baseband_signal)
        
        if use_rrc:
            baseband_signal = self._apply_matched_filter(baseband_signal)
        
        symbols = self._sample_symbols(baseband_signal)
        
        # Calcula LLRs (simplificado - assume AWGN)
        num_symbols = len(symbols)
        llrs = np.zeros(num_symbols * 4, dtype=float)
        
        noise_var = self._estimate_noise_variance(symbols)
        
        for i, rx_symbol in enumerate(symbols):
            for bit_pos in range(4):
                # Símbolos com bit=0 nesta posição
                mask_0 = [(idx >> (3 - bit_pos)) & 1 == 0 for idx in range(16)]
                symbols_0 = self.constellation[mask_0]
                
                # Símbolos com bit=1 nesta posição
                mask_1 = [(idx >> (3 - bit_pos)) & 1 == 1 for idx in range(16)]
                symbols_1 = self.constellation[mask_1]
                
                # Log-likelihood ratio
                dist_0 = np.min(np.abs(rx_symbol - symbols_0)**2)
                dist_1 = np.min(np.abs(rx_symbol - symbols_1)**2)
                
                llrs[i*4 + bit_pos] = (dist_1 - dist_0) / (2 * noise_var)
        
        return llrs
    
    def _estimate_noise_variance(self, symbols: np.ndarray) -> float:
        """
        Estima variância do ruído usando símbolos recebidos.
        
        Args:
            symbols: Símbolos recebidos
            
        Returns:
            Estimativa da variância do ruído
        """
        # Decisão hard para cada símbolo
        decided_symbols = np.zeros_like(symbols)
        
        for i, symbol in enumerate(symbols):
            distances = np.abs(symbol - self.constellation)
            min_idx = np.argmin(distances)
            decided_symbols[i] = self.constellation[min_idx]
        
        # Erro = ruído + ISI (aproximadamente)
        error = symbols - decided_symbols
        noise_var = np.var(error)
        
        return max(noise_var, 1e-10)  # Evita divisão por zero
    
    def receive_text(self, signal: np.ndarray, encoding: str = 'utf-8', 
                     use_rrc: bool = False) -> str:
        """
        Recebe e decodifica texto.
        
        Args:
            signal: Sinal recebido
            encoding: Codificação de caracteres
            use_rrc: Se True, usa filtro casado RRC
            
        Returns:
            Texto decodificado
        """
        bits = self.demodulate(signal, use_rrc=use_rrc)
        data_bytes = self.bytes_from_bits(bits)
        
        try:
            text = data_bytes.decode(encoding, errors='ignore')
            return text
        except Exception as e:
            return f"[Erro na decodificação: {e}]"
    
    def plot_constellation_diagram(self, signal: np.ndarray, use_rrc: bool = False, 
                                   max_symbols: int = 1000):
        """
        Plota diagrama de constelação dos símbolos recebidos.
        
        Args:
            signal: Sinal recebido
            use_rrc: Se True, aplica filtro casado
            max_symbols: Número máximo de símbolos a plotar
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível")
            return
        
        # Processa sinal
        baseband_signal = self.downconvert(signal)
        baseband_signal = self.apply_agc(baseband_signal)
        
        if use_rrc:
            baseband_signal = self._apply_matched_filter(baseband_signal)
        
        symbols = self._sample_symbols(baseband_signal)
        symbols = symbols[:max_symbols]
        
        # Plota
        plt.figure(figsize=(10, 10))
        
        # Símbolos recebidos
        plt.scatter(np.real(symbols), np.imag(symbols), 
                   s=20, c='blue', alpha=0.5, label='Símbolos recebidos')
        
        # Constelação ideal
        plt.scatter(np.real(self.constellation), np.imag(self.constellation), 
                   s=200, c='red', marker='x', linewidths=3, label='Constelação ideal')
        
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.title(f'Diagrama de Constelação 16-QAM ({len(symbols)} símbolos)')
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_bit_rate(self) -> float:
        """
        Retorna a taxa de bits em bps.
        
        Returns:
            Taxa de bits (bps)
        """
        return self.symbol_rate * 4  # 16-QAM: 4 bits/símbolo
    
    def __str__(self) -> str:
        return (f"16-QAM Receiver:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Symbol Rate: {self.symbol_rate/1e3:.2f} kbaud\n"
                f"  Bit Rate: {self.get_bit_rate()/1e3:.2f} kbps\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz\n"
                f"  Samples/Symbol: {self.samples_per_symbol}")
