"""
Transmissor OFDM (Orthogonal Frequency-Division Multiplexing).
Modulação multiportadora onde dados são distribuídos em múltiplas subportadoras ortogonais.
"""
import numpy as np
from .base_transmitter import BaseTransmitter
from typing import Optional


class OFDMTransmitter(BaseTransmitter):
    """
    Transmissor OFDM (Orthogonal Frequency-Division Multiplexing).
    
    OFDM divide o espectro em múltiplas subportadoras ortogonais:
    - Cada subportadora pode usar modulação diferente (BPSK, QPSK, QAM)
    - Resistente a multipercurso e desvanecimento seletivo em frequência
    - Eficiente espectralmente
    - Usa IFFT para modulação e FFT para demodulação
    
    Parâmetros principais:
    - num_subcarriers: Número de subportadoras (tamanho da FFT)
    - cp_length: Comprimento do prefixo cíclico
    - subcarrier_modulation: Tipo de modulação por subportadora
    """
    
    def __init__(self, sample_rate: float, num_subcarriers: int = 64, 
                 cp_length: int = 16, subcarrier_modulation: str = 'QPSK',
                 carrier_freq: float = 0):
        """
        Inicializa o transmissor OFDM.
        
        Args:
            sample_rate: Taxa de amostragem em Hz
            num_subcarriers: Número de subportadoras (tamanho da FFT)
            cp_length: Comprimento do prefixo cíclico (em amostras)
            subcarrier_modulation: Tipo de modulação ('BPSK', 'QPSK', '16QAM')
            carrier_freq: Frequência da portadora em Hz (0 para banda base)
        """
        super().__init__(sample_rate, carrier_freq)
        
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.subcarrier_modulation = subcarrier_modulation.upper()
        
        # Número de subportadoras de dados (excluindo DC e guardas)
        self.num_data_subcarriers = num_subcarriers - 2  # Remove DC e Nyquist
        
        # Tamanho do símbolo OFDM (FFT + CP)
        self.ofdm_symbol_size = num_subcarriers + cp_length
        self.samples_per_symbol = self.ofdm_symbol_size
        
        # Taxa de símbolos OFDM
        self.symbol_rate = sample_rate / self.ofdm_symbol_size
        
        # Bits por subportadora (depende da modulação)
        self.bits_per_subcarrier = self._get_bits_per_subcarrier()
        
        # Bits por símbolo OFDM
        self.bits_per_ofdm_symbol = self.num_data_subcarriers * self.bits_per_subcarrier
        
        # Gera constelação para subportadoras
        self.constellation = self._generate_constellation()
        
    def _get_bits_per_subcarrier(self) -> int:
        """
        Retorna número de bits por subportadora baseado na modulação.
        
        Returns:
            Bits por subportadora
        """
        if self.subcarrier_modulation == 'BPSK':
            return 1
        elif self.subcarrier_modulation == 'QPSK':
            return 2
        elif self.subcarrier_modulation in ['16QAM', 'QAM16']:
            return 4
        elif self.subcarrier_modulation in ['64QAM', 'QAM64']:
            return 6
        else:
            raise ValueError(f"Modulação '{self.subcarrier_modulation}' não suportada")
    
    def _generate_constellation(self) -> np.ndarray:
        """
        Gera constelação para modulação das subportadoras.
        
        Returns:
            Array com pontos da constelação
        """
        if self.subcarrier_modulation == 'BPSK':
            # BPSK: -1, +1
            constellation = np.array([-1, 1], dtype=np.complex64)
            
        elif self.subcarrier_modulation == 'QPSK':
            # QPSK: 4 pontos (Gray-coded)
            constellation = np.array([
                1+1j, -1+1j, -1-1j, 1-1j
            ], dtype=np.complex64) / np.sqrt(2)
            
        elif self.subcarrier_modulation in ['16QAM', 'QAM16']:
            # 16-QAM
            levels = np.array([-3, -1, 1, 3])
            constellation = np.zeros(16, dtype=np.complex64)
            
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
            
            avg_power = np.mean(np.abs(constellation)**2)
            constellation = constellation / np.sqrt(avg_power)
            
        else:
            raise ValueError(f"Modulação '{self.subcarrier_modulation}' não implementada")
        
        return constellation
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modula bits usando OFDM.
        
        Args:
            bits: Array de bits (0s e 1s)
            
        Returns:
            Sinal complexo I/Q modulado
        """
        # Garante que temos bits suficientes para preencher símbolos OFDM completos
        bits_per_symbol = self.bits_per_ofdm_symbol
        
        if len(bits) % bits_per_symbol != 0:
            # Padding com zeros
            padding = bits_per_symbol - (len(bits) % bits_per_symbol)
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])
        
        # Número de símbolos OFDM
        num_ofdm_symbols = len(bits) // bits_per_symbol
        
        # Lista para armazenar símbolos OFDM modulados
        ofdm_signal = []
        
        for sym_idx in range(num_ofdm_symbols):
            # Extrai bits para este símbolo OFDM
            start_bit = sym_idx * bits_per_symbol
            end_bit = start_bit + bits_per_symbol
            symbol_bits = bits[start_bit:end_bit]
            
            # Modula subportadoras
            subcarrier_symbols = self._modulate_subcarriers(symbol_bits)
            
            # Monta símbolo OFDM no domínio da frequência
            freq_domain = self._map_to_subcarriers(subcarrier_symbols)
            
            # IFFT para domínio do tempo
            time_domain = np.fft.ifft(freq_domain)
            
            # Adiciona prefixo cíclico
            ofdm_symbol = self._add_cyclic_prefix(time_domain)
            
            ofdm_signal.append(ofdm_symbol)
        
        # Concatena todos os símbolos OFDM
        baseband_signal = np.concatenate(ofdm_signal)
        
        # Normaliza
        baseband_signal = self.normalize_signal(baseband_signal)
        
        # Upconvert para portadora (se necessário)
        rf_signal = self.upconvert(baseband_signal)
        
        return rf_signal.astype(np.complex64)
    
    def _modulate_subcarriers(self, bits: np.ndarray) -> np.ndarray:
        """
        Modula bits em símbolos de subportadoras.
        
        Args:
            bits: Bits para este símbolo OFDM
            
        Returns:
            Símbolos complexos para cada subportadora de dados
        """
        num_data_symbols = len(bits) // self.bits_per_subcarrier
        symbols = np.zeros(num_data_symbols, dtype=np.complex64)
        
        for i in range(num_data_symbols):
            # Extrai bits para esta subportadora
            start = i * self.bits_per_subcarrier
            end = start + self.bits_per_subcarrier
            bit_group = bits[start:end]
            
            # Converte para índice
            symbol_idx = 0
            for j, bit in enumerate(bit_group):
                symbol_idx |= (bit << (self.bits_per_subcarrier - 1 - j))
            
            # Mapeia para constelação
            symbols[i] = self.constellation[symbol_idx]
        
        return symbols
    
    def _map_to_subcarriers(self, symbols: np.ndarray) -> np.ndarray:
        """
        Mapeia símbolos de dados para subportadoras FFT.
        
        Args:
            symbols: Símbolos de dados modulados
            
        Returns:
            Array do tamanho da FFT com símbolos mapeados
        """
        freq_domain = np.zeros(self.num_subcarriers, dtype=np.complex64)
        
        # Subportadora DC (índice 0) = 0
        freq_domain[0] = 0
        
        # Mapeia símbolos de dados
        # Metade inferior do espectro (subportadoras positivas)
        num_positive = len(symbols) // 2
        freq_domain[1:1+num_positive] = symbols[:num_positive]
        
        # Metade superior do espectro (subportadoras negativas)
        num_negative = len(symbols) - num_positive
        freq_domain[-num_negative:] = symbols[num_positive:]
        
        return freq_domain
    
    def _add_cyclic_prefix(self, ofdm_symbol: np.ndarray) -> np.ndarray:
        """
        Adiciona prefixo cíclico ao símbolo OFDM.
        
        O prefixo cíclico copia as últimas amostras do símbolo
        para o início, criando periodicidade circular.
        
        Args:
            ofdm_symbol: Símbolo OFDM no domínio do tempo
            
        Returns:
            Símbolo OFDM com prefixo cíclico
        """
        # Copia últimas cp_length amostras para o início
        cp = ofdm_symbol[-self.cp_length:]
        symbol_with_cp = np.concatenate([cp, ofdm_symbol])
        
        return symbol_with_cp
    
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
        return self.symbol_rate * self.bits_per_ofdm_symbol
    
    def get_spectral_efficiency(self) -> float:
        """
        Retorna a eficiência espectral em bits/s/Hz.
        
        Returns:
            Eficiência espectral
        """
        # Eficiência considerando overhead do CP
        efficiency = self.bits_per_ofdm_symbol / self.ofdm_symbol_size
        return efficiency * self.sample_rate / self.num_subcarriers
    
    def get_bandwidth(self) -> float:
        """
        Retorna a largura de banda ocupada.
        
        Returns:
            Largura de banda em Hz
        """
        return self.sample_rate
    
    def plot_spectrum(self, signal: Optional[np.ndarray] = None):
        """
        Plota o espectro de frequência do sinal OFDM.
        
        Args:
            signal: Sinal a analisar (opcional, gera um se None)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível")
            return
        
        if signal is None:
            # Gera sinal de teste
            test_bits = np.random.randint(0, 2, self.bits_per_ofdm_symbol * 10)
            signal = self.modulate(test_bits)
        
        # Calcula FFT
        N = len(signal)
        fft_signal = np.fft.fftshift(np.fft.fft(signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/self.sample_rate))
        
        # Potência em dB
        power_db = 20 * np.log10(np.abs(fft_signal) + 1e-10)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(freqs/1e6, power_db)
        plt.xlabel('Frequência (MHz)', fontsize=12)
        plt.ylabel('Potência (dB)', fontsize=12)
        plt.title('Espectro de Potência OFDM', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Zoom nas subportadoras centrais
        plt.subplot(1, 2, 2)
        center_idx = N // 2
        zoom_range = self.num_subcarriers * 2
        start_idx = max(0, center_idx - zoom_range)
        end_idx = min(N, center_idx + zoom_range)
        
        plt.plot(freqs[start_idx:end_idx]/1e3, power_db[start_idx:end_idx])
        plt.xlabel('Frequência (kHz)', fontsize=12)
        plt.ylabel('Potência (dB)', fontsize=12)
        plt.title('Espectro - Zoom nas Subportadoras', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        return (f"OFDM Transmitter:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Num Subcarriers: {self.num_subcarriers}\n"
                f"  Data Subcarriers: {self.num_data_subcarriers}\n"
                f"  CP Length: {self.cp_length}\n"
                f"  Subcarrier Modulation: {self.subcarrier_modulation}\n"
                f"  Bits/Subcarrier: {self.bits_per_subcarrier}\n"
                f"  Bits/OFDM Symbol: {self.bits_per_ofdm_symbol}\n"
                f"  OFDM Symbol Rate: {self.symbol_rate/1e3:.2f} ksym/s\n"
                f"  Bit Rate: {self.get_bit_rate()/1e6:.3f} Mbps\n"
                f"  Spectral Efficiency: {self.get_spectral_efficiency():.2f} bits/s/Hz\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz")
