"""
Transmissor BPSK (Binary Phase Shift Keying).
Modulação digital onde cada bit é representado por uma fase (0° ou 180°).
"""
import numpy as np
from .base_transmitter import BaseTransmitter


class BPSKTransmitter(BaseTransmitter):
    """
    Transmissor BPSK (Binary Phase Shift Keying).
    
    Na modulação BPSK:
    - Bit 0 -> Fase 180° (símbolo -1)
    - Bit 1 -> Fase 0° (símbolo +1)
    
    Taxa de transmissão: 1 bit por símbolo
    """
    
    def __init__(self, sample_rate: float, symbol_rate: float, carrier_freq: float = 0):
        """
        Inicializa o transmissor BPSK.
        
        Args:
            sample_rate: Taxa de amostragem em Hz (ex: 1e6 para 1 MS/s)
            symbol_rate: Taxa de símbolos em Hz (ex: 1e3 para 1 kbaud)
            carrier_freq: Frequência da portadora em Hz (0 para banda base)
        """
        super().__init__(sample_rate, carrier_freq)
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        
        if self.samples_per_symbol < 2:
            raise ValueError(f"Taxa de amostragem muito baixa. Necessário pelo menos 2 amostras/símbolo. "
                           f"Atual: {self.samples_per_symbol}")
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modula bits usando BPSK.
        
        Args:
            bits: Array de bits (0s e 1s)
            
        Returns:
            Sinal complexo I/Q modulado
        """
        # Converte bits para símbolos BPSK: 0 -> -1, 1 -> +1
        symbols = 2 * bits - 1
        
        # Gera forma de pulso (Raised Cosine ou retangular)
        baseband_signal = self._pulse_shape(symbols)
        
        # Normaliza
        baseband_signal = self.normalize_signal(baseband_signal)
        
        # Upconvert para portadora (se necessário)
        rf_signal = self.upconvert(baseband_signal)
        
        return rf_signal.astype(np.complex64)
    
    def _pulse_shape(self, symbols: np.ndarray, pulse_type: str = 'rect') -> np.ndarray:
        """
        Aplica formatação de pulso aos símbolos.
        
        Args:
            symbols: Array de símbolos (+1 ou -1)
            pulse_type: Tipo de pulso ('rect' para retangular, 'rrc' para root raised cosine)
            
        Returns:
            Sinal em banda base com pulsos formatados
        """
        if pulse_type == 'rect':
            # Pulso retangular simples (repete cada símbolo samples_per_symbol vezes)
            signal = np.repeat(symbols, self.samples_per_symbol)
            
        elif pulse_type == 'rrc':
            # Root Raised Cosine (mais realista, reduz ISI)
            signal = self._apply_rrc_filter(symbols)
        else:
            raise ValueError(f"Tipo de pulso '{pulse_type}' não suportado")
        
        # Para BPSK, o sinal é real, mas retornamos como complexo para consistência
        return signal.astype(np.complex64)
    
    def _apply_rrc_filter(self, symbols: np.ndarray, alpha: float = 0.35, span: int = 10) -> np.ndarray:
        """
        Aplica filtro Root Raised Cosine (RRC).
        
        Args:
            symbols: Símbolos a serem filtrados
            alpha: Fator de roll-off (0 < alpha <= 1)
            span: Número de símbolos do filtro (maior = melhor, mas mais lento)
            
        Returns:
            Sinal filtrado
        """
        # Upsample: insere zeros entre símbolos
        upsampled = np.zeros(len(symbols) * self.samples_per_symbol)
        upsampled[::self.samples_per_symbol] = symbols
        
        # Gera filtro RRC
        rrc_filter = self._rrc_coefficients(alpha, span)
        
        # Convolução
        signal = np.convolve(upsampled, rrc_filter, mode='same')
        
        return signal
    
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
        
        # Evita divisão por zero
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
        
        # Normaliza energia
        h = h / np.sqrt(np.sum(h**2))
        
        return h
    
    def transmit_text(self, text: str, encoding: str = 'utf-8') -> np.ndarray:
        """
        Transmite texto codificado.
        
        Args:
            text: Texto a transmitir
            encoding: Codificação de caracteres
            
        Returns:
            Sinal I/Q modulado
        """
        data_bytes = text.encode(encoding)
        bits = self.bits_from_bytes(data_bytes)
        return self.modulate(bits)
    
    def get_bit_rate(self) -> float:
        """
        Retorna a taxa de bits em bps.
        
        Returns:
            Taxa de bits (bps)
        """
        return self.symbol_rate  # Para BPSK, 1 bit/símbolo
    
    def get_spectral_efficiency(self) -> float:
        """
        Retorna a eficiência espectral em bits/s/Hz.
        
        Returns:
            Eficiência espectral
        """
        return 1.0  # BPSK: 1 bit/s/Hz (teórico)
    
    def __str__(self) -> str:
        return (f"BPSK Transmitter:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Symbol Rate: {self.symbol_rate/1e3:.2f} kbaud\n"
                f"  Bit Rate: {self.get_bit_rate()/1e3:.2f} kbps\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz\n"
                f"  Samples/Symbol: {self.samples_per_symbol}")
