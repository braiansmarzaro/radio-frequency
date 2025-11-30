"""
Transmissor 4-ASK (4-level Amplitude Shift Keying).
Modulação digital onde cada símbolo representa 2 bits usando 4 níveis de amplitude.
"""
import numpy as np
from .base_transmitter import BaseTransmitter


class ASK4Transmitter(BaseTransmitter):
    """
    Transmissor 4-ASK (4-level Amplitude Shift Keying).
    
    Na modulação 4-ASK:
    - Cada símbolo representa 2 bits
    - 4 níveis de amplitude diferentes
    - Fase constante (apenas amplitude varia)
    
    Taxa de transmissão: 2 bits por símbolo
    Eficiência espectral: 2 bits/s/Hz (teórico)
    """
    
    def __init__(self, sample_rate: float, symbol_rate: float, carrier_freq: float = 0):
        """
        Inicializa o transmissor 4-ASK.
        
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
        
        # Gera níveis de amplitude 4-ASK (normalizado)
        self.amplitude_levels = self._generate_amplitude_levels()
    
    def _generate_amplitude_levels(self) -> np.ndarray:
        """
        Gera os 4 níveis de amplitude para 4-ASK.
        
        Níveis: [-3, -1, +1, +3] normalizados
        Mapeamento Gray-coded:
        00 -> -3
        01 -> -1
        11 -> +1
        10 -> +3
        
        Returns:
            Array com 4 níveis de amplitude
        """
        # Níveis equidistantes
        levels = np.array([-3, -1, 1, 3], dtype=float)
        
        # Normaliza energia média para 1
        avg_power = np.mean(levels**2)
        levels = levels / np.sqrt(avg_power)
        
        return levels
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modula bits usando 4-ASK.
        
        Args:
            bits: Array de bits (0s e 1s). Comprimento deve ser múltiplo de 2.
            
        Returns:
            Sinal complexo I/Q modulado (Q=0 para ASK)
        """
        # Garante que o número de bits é múltiplo de 2
        if len(bits) % 2 != 0:
            # Padding com zero
            bits = np.concatenate([bits, np.zeros(1, dtype=int)])
        
        # Agrupa bits em símbolos de 2 bits (Gray coding)
        num_symbols = len(bits) // 2
        symbols = np.zeros(num_symbols, dtype=float)
        
        # Mapeamento Gray-coded
        gray_map = {
            (0, 0): 0,  # -3
            (0, 1): 1,  # -1
            (1, 1): 2,  # +1
            (1, 0): 3   # +3
        }
        
        for i in range(num_symbols):
            bit_pair = (bits[i*2], bits[i*2 + 1])
            level_idx = gray_map[tuple(bit_pair)]
            symbols[i] = self.amplitude_levels[level_idx]
        
        # Aplica formatação de pulso
        baseband_signal = self._pulse_shape(symbols)
        
        # Normaliza
        baseband_signal = self.normalize_signal(baseband_signal)
        
        # Upconvert para portadora (se necessário)
        # ASK é real, mas convertemos para complexo para consistência
        rf_signal = self.upconvert(baseband_signal.astype(np.complex64))
        
        return rf_signal.astype(np.complex64)
    
    def _pulse_shape(self, symbols: np.ndarray, pulse_type: str = 'rect') -> np.ndarray:
        """
        Aplica formatação de pulso aos símbolos.
        
        Args:
            symbols: Array de símbolos (amplitudes)
            pulse_type: Tipo de pulso ('rect' para retangular, 'rrc' para root raised cosine)
            
        Returns:
            Sinal em banda base com pulsos formatados
        """
        if pulse_type == 'rect':
            # Pulso retangular simples
            signal = np.repeat(symbols, self.samples_per_symbol)
            
        elif pulse_type == 'rrc':
            # Root Raised Cosine
            signal = self._apply_rrc_filter(symbols)
        else:
            raise ValueError(f"Tipo de pulso '{pulse_type}' não suportado")
        
        return signal
    
    def _apply_rrc_filter(self, symbols: np.ndarray, alpha: float = 0.35, span: int = 10) -> np.ndarray:
        """
        Aplica filtro Root Raised Cosine (RRC).
        
        Args:
            symbols: Símbolos a serem filtrados
            alpha: Fator de roll-off (0 < alpha <= 1)
            span: Número de símbolos do filtro
            
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
        return self.symbol_rate * 2  # 4-ASK: 2 bits/símbolo
    
    def get_spectral_efficiency(self) -> float:
        """
        Retorna a eficiência espectral em bits/s/Hz.
        
        Returns:
            Eficiência espectral
        """
        return 2.0  # 4-ASK: 2 bits/s/Hz (teórico)
    
    def plot_signal_levels(self):
        """
        Plota os níveis de amplitude do 4-ASK.
        
        Requer matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível para plotagem")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plota níveis de amplitude
        bit_patterns = ['00', '01', '11', '10']
        positions = [0, 1, 2, 3]
        
        plt.stem(positions, self.amplitude_levels, basefmt=' ')
        
        # Adiciona labels
        for i, (pos, level, bits) in enumerate(zip(positions, self.amplitude_levels, bit_patterns)):
            plt.annotate(f'{bits}\n({level:.3f})', 
                        xy=(pos, level), 
                        xytext=(0, 10 if level > 0 else -20),
                        textcoords='offset points',
                        ha='center',
                        fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.xlabel('Índice do Símbolo', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title('Níveis de Amplitude 4-ASK (Gray-coded)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(positions, bit_patterns)
        plt.tight_layout()
        plt.show()
    
    def plot_eye_diagram(self, bits: np.ndarray, num_traces: int = 50):
        """
        Plota diagrama de olho do sinal 4-ASK.
        
        Args:
            bits: Bits a modular para o diagrama
            num_traces: Número de traços a plotar
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível")
            return
        
        # Modula os bits
        signal = self.modulate(bits)
        signal = np.real(signal)  # ASK é real
        
        # Prepara diagrama de olho
        samples_per_trace = self.samples_per_symbol * 2
        num_traces = min(num_traces, len(signal) // samples_per_trace)
        
        plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            start = i * self.samples_per_symbol
            end = start + samples_per_trace
            
            if end <= len(signal):
                trace = signal[start:end]
                t = np.arange(len(trace)) / self.sample_rate * 1e3  # ms
                plt.plot(t, trace, 'b', alpha=0.3, linewidth=0.5)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Tempo (ms)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title('Diagrama de Olho - 4-ASK', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        return (f"4-ASK Transmitter:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Symbol Rate: {self.symbol_rate/1e3:.2f} kbaud\n"
                f"  Bit Rate: {self.get_bit_rate()/1e3:.2f} kbps\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz\n"
                f"  Samples/Symbol: {self.samples_per_symbol}\n"
                f"  Spectral Efficiency: {self.get_spectral_efficiency()} bits/s/Hz\n"
                f"  Amplitude Levels: {self.amplitude_levels}")
