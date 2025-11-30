"""
Transmissor 16-QAM (16-Quadrature Amplitude Modulation).
Modulação digital onde cada símbolo representa 4 bits usando 16 pontos na constelação.
"""
import numpy as np
from .base_transmitter import BaseTransmitter


class QAM16Transmitter(BaseTransmitter):
    """
    Transmissor 16-QAM (16-Quadrature Amplitude Modulation).
    
    Na modulação 16-QAM:
    - Cada símbolo representa 4 bits
    - 16 pontos na constelação (grade 4x4)
    - Combina modulação de amplitude e fase
    
    Taxa de transmissão: 4 bits por símbolo
    Eficiência espectral: 4 bits/s/Hz (teórico)
    """
    
    def __init__(self, sample_rate: float, symbol_rate: float, carrier_freq: float = 0):
        """
        Inicializa o transmissor 16-QAM.
        
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
        
        # Gera mapa de constelação 16-QAM (normalizado)
        self.constellation = self._generate_constellation()
    
    def _generate_constellation(self) -> np.ndarray:
        """
        Gera a constelação 16-QAM (grade 4x4).
        
        Constelação Gray-coded para minimizar BER:
        Níveis I e Q: [-3, -1, +1, +3]
        
        Returns:
            Array com 16 símbolos complexos
        """
        # Níveis de amplitude (normalizados)
        levels = np.array([-3, -1, 1, 3])
        
        # Cria constelação 16-QAM (4x4 grid)
        constellation = np.zeros(16, dtype=np.complex64)
        
        # Mapeamento Gray-coded (minimiza erros de bit)
        # Índice binário -> símbolo complexo
        gray_map = [
            0b0000, 0b0001, 0b0011, 0b0010,  # Linha 1
            0b0100, 0b0101, 0b0111, 0b0110,  # Linha 2
            0b1100, 0b1101, 0b1111, 0b1110,  # Linha 3
            0b1000, 0b1001, 0b1011, 0b1010   # Linha 4
        ]
        
        idx = 0
        for i in range(4):
            for q in range(4):
                gray_idx = gray_map[idx]
                constellation[gray_idx] = levels[q] + 1j * levels[3-i]
                idx += 1
        
        # Normaliza energia média para 1
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        
        return constellation
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modula bits usando 16-QAM.
        
        Args:
            bits: Array de bits (0s e 1s). Comprimento deve ser múltiplo de 4.
            
        Returns:
            Sinal complexo I/Q modulado
        """
        # Garante que o número de bits é múltiplo de 4
        if len(bits) % 4 != 0:
            # Padding com zeros
            padding = 4 - (len(bits) % 4)
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        # Agrupa bits em símbolos de 4 bits
        num_symbols = len(bits) // 4
        symbols = np.zeros(num_symbols, dtype=np.complex64)
        
        for i in range(num_symbols):
            # Extrai 4 bits
            bit_group = bits[i*4:(i+1)*4]
            
            # Converte para índice (0-15)
            symbol_idx = (bit_group[0] << 3) | (bit_group[1] << 2) | \
                        (bit_group[2] << 1) | bit_group[3]
            
            # Mapeia para símbolo da constelação
            symbols[i] = self.constellation[symbol_idx]
        
        # Aplica formatação de pulso
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
            symbols: Array de símbolos complexos
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
        
        return signal.astype(np.complex64)
    
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
        upsampled = np.zeros(len(symbols) * self.samples_per_symbol, dtype=np.complex64)
        upsampled[::self.samples_per_symbol] = symbols
        
        # Gera filtro RRC
        rrc_filter = self._rrc_coefficients(alpha, span)
        
        # Convolução (separada para I e Q)
        signal_i = np.convolve(np.real(upsampled), rrc_filter, mode='same')
        signal_q = np.convolve(np.imag(upsampled), rrc_filter, mode='same')
        
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
        return self.symbol_rate * 4  # 16-QAM: 4 bits/símbolo
    
    def get_spectral_efficiency(self) -> float:
        """
        Retorna a eficiência espectral em bits/s/Hz.
        
        Returns:
            Eficiência espectral
        """
        return 4.0  # 16-QAM: 4 bits/s/Hz (teórico)
    
    def plot_constellation(self):
        """
        Plota o diagrama de constelação 16-QAM.
        
        Requer matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível para plotagem")
            return
        
        plt.figure(figsize=(8, 8))
        
        # Plota pontos da constelação
        plt.scatter(np.real(self.constellation), np.imag(self.constellation), 
                   s=100, c='blue', marker='o', label='Símbolos')
        
        # Adiciona labels com valores binários
        for idx, point in enumerate(self.constellation):
            binary = format(idx, '04b')
            plt.annotate(binary, (np.real(point), np.imag(point)), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.title('Constelação 16-QAM (Gray-coded)')
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        return (f"16-QAM Transmitter:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Symbol Rate: {self.symbol_rate/1e3:.2f} kbaud\n"
                f"  Bit Rate: {self.get_bit_rate()/1e3:.2f} kbps\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz\n"
                f"  Samples/Symbol: {self.samples_per_symbol}\n"
                f"  Spectral Efficiency: {self.get_spectral_efficiency()} bits/s/Hz")
