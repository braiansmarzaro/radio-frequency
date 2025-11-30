"""
Receptor OFDM (Orthogonal Frequency-Division Multiplexing).
Demodula sinais OFDM multiportadora com FFT.
"""
import numpy as np
from .base_receiver import BaseReceiver
from typing import Optional


class OFDMReceiver(BaseReceiver):
    """
    Receptor OFDM (Orthogonal Frequency-Division Multiplexing).
    
    Demodula sinais OFDM:
    - Remove prefixo cíclico
    - Aplica FFT para obter símbolos no domínio da frequência
    - Demodula cada subportadora
    - Detecta símbolos por mínima distância
    
    Deve ser pareado com OFDMTransmitter com mesmos parâmetros.
    """
    
    def __init__(self, sample_rate: float, num_subcarriers: int = 64,
                 cp_length: int = 16, subcarrier_modulation: str = 'QPSK',
                 carrier_freq: float = 0):
        """
        Inicializa o receptor OFDM.
        
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
        
        # Número de subportadoras de dados
        self.num_data_subcarriers = num_subcarriers - 2
        
        # Tamanho do símbolo OFDM
        self.ofdm_symbol_size = num_subcarriers + cp_length
        self.samples_per_symbol = self.ofdm_symbol_size
        
        # Taxa de símbolos OFDM
        self.symbol_rate = sample_rate / self.ofdm_symbol_size
        
        # Bits por subportadora
        self.bits_per_subcarrier = self._get_bits_per_subcarrier()
        
        # Bits por símbolo OFDM
        self.bits_per_ofdm_symbol = self.num_data_subcarriers * self.bits_per_subcarrier
        
        # Gera constelação
        self.constellation = self._generate_constellation()
        
    def _get_bits_per_subcarrier(self) -> int:
        """Retorna número de bits por subportadora baseado na modulação."""
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
        """Gera constelação para modulação das subportadoras."""
        if self.subcarrier_modulation == 'BPSK':
            constellation = np.array([-1, 1], dtype=np.complex64)
            
        elif self.subcarrier_modulation == 'QPSK':
            constellation = np.array([
                1+1j, -1+1j, -1-1j, 1-1j
            ], dtype=np.complex64) / np.sqrt(2)
            
        elif self.subcarrier_modulation in ['16QAM', 'QAM16']:
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
    
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """
        Demodula sinal OFDM recebido.
        
        Args:
            signal: Sinal complexo I/Q recebido
            
        Returns:
            Array de bits demodulados (0s e 1s)
        """
        # Downconvert do RF para banda base (se necessário)
        baseband_signal = self.downconvert(signal)
        
        # Aplica AGC
        baseband_signal = self.apply_agc(baseband_signal)
        
        # Número de símbolos OFDM completos
        num_ofdm_symbols = len(baseband_signal) // self.ofdm_symbol_size
        
        # Lista para armazenar bits demodulados
        all_bits = []
        
        for sym_idx in range(num_ofdm_symbols):
            # Extrai símbolo OFDM
            start_idx = sym_idx * self.ofdm_symbol_size
            end_idx = start_idx + self.ofdm_symbol_size
            ofdm_symbol = baseband_signal[start_idx:end_idx]
            
            # Remove prefixo cíclico
            time_domain = self._remove_cyclic_prefix(ofdm_symbol)
            
            # FFT para domínio da frequência
            freq_domain = np.fft.fft(time_domain)
            
            # Extrai símbolos das subportadoras de dados
            data_symbols = self._extract_data_subcarriers(freq_domain)
            
            # Demodula subportadoras
            symbol_bits = self._demodulate_subcarriers(data_symbols)
            
            all_bits.extend(symbol_bits)
        
        return np.array(all_bits, dtype=int)
    
    def _remove_cyclic_prefix(self, ofdm_symbol: np.ndarray) -> np.ndarray:
        """
        Remove prefixo cíclico do símbolo OFDM.
        
        Args:
            ofdm_symbol: Símbolo OFDM com prefixo cíclico
            
        Returns:
            Símbolo OFDM sem prefixo cíclico
        """
        # Remove as primeiras cp_length amostras
        return ofdm_symbol[self.cp_length:]
    
    def _extract_data_subcarriers(self, freq_domain: np.ndarray) -> np.ndarray:
        """
        Extrai símbolos das subportadoras de dados.
        
        Args:
            freq_domain: Símbolos no domínio da frequência (saída da FFT)
            
        Returns:
            Símbolos das subportadoras de dados
        """
        data_symbols = np.zeros(self.num_data_subcarriers, dtype=np.complex64)
        
        # Metade inferior (subportadoras positivas)
        num_positive = self.num_data_subcarriers // 2
        data_symbols[:num_positive] = freq_domain[1:1+num_positive]
        
        # Metade superior (subportadoras negativas)
        num_negative = self.num_data_subcarriers - num_positive
        data_symbols[num_positive:] = freq_domain[-num_negative:]
        
        return data_symbols
    
    def _demodulate_subcarriers(self, symbols: np.ndarray) -> np.ndarray:
        """
        Demodula símbolos das subportadoras.
        
        Args:
            symbols: Símbolos complexos das subportadoras
            
        Returns:
            Bits demodulados
        """
        bits = []
        
        for symbol in symbols:
            # Encontra símbolo mais próximo na constelação
            distances = np.abs(self.constellation - symbol)
            symbol_idx = np.argmin(distances)
            
            # Converte índice para bits
            symbol_bits = self._symbol_to_bits(symbol_idx)
            bits.extend(symbol_bits)
        
        return np.array(bits, dtype=int)
    
    def _symbol_to_bits(self, symbol_idx: int) -> np.ndarray:
        """
        Converte índice do símbolo para bits.
        
        Args:
            symbol_idx: Índice na constelação
            
        Returns:
            Array de bits
        """
        bits = np.zeros(self.bits_per_subcarrier, dtype=int)
        
        for i in range(self.bits_per_subcarrier):
            bit_position = self.bits_per_subcarrier - 1 - i
            bits[i] = (symbol_idx >> bit_position) & 1
        
        return bits
    
    def receive_text(self, signal: np.ndarray, encoding: str = 'utf-8',
                    original_length: Optional[int] = None) -> str:
        """
        Recebe e decodifica texto.
        
        Args:
            signal: Sinal I/Q recebido
            encoding: Codificação de caracteres
            original_length: Comprimento original em bytes (para remover padding)
            
        Returns:
            Texto decodificado
        """
        bits = self.demodulate(signal)
        data_bytes = self.bytes_from_bits(bits)
        
        if original_length is not None:
            data_bytes = data_bytes[:original_length]
        
        try:
            text = data_bytes.decode(encoding)
            return text
        except UnicodeDecodeError:
            # Tenta ignorar erros de decodificação
            return data_bytes.decode(encoding, errors='ignore')
    
    def plot_constellation(self, signal: np.ndarray, num_symbols: int = 500):
        """
        Plota diagrama de constelação dos símbolos recebidos.
        
        Args:
            signal: Sinal recebido
            num_symbols: Número de símbolos a plotar
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível")
            return
        
        # Demodula parcialmente para obter símbolos
        baseband_signal = self.downconvert(signal)
        baseband_signal = self.apply_agc(baseband_signal)
        
        # Extrai alguns símbolos OFDM
        num_ofdm_symbols = min(
            num_symbols // self.num_data_subcarriers + 1,
            len(baseband_signal) // self.ofdm_symbol_size
        )
        
        received_symbols = []
        
        for sym_idx in range(num_ofdm_symbols):
            start_idx = sym_idx * self.ofdm_symbol_size
            end_idx = start_idx + self.ofdm_symbol_size
            
            if end_idx > len(baseband_signal):
                break
            
            ofdm_symbol = baseband_signal[start_idx:end_idx]
            time_domain = self._remove_cyclic_prefix(ofdm_symbol)
            freq_domain = np.fft.fft(time_domain)
            data_symbols = self._extract_data_subcarriers(freq_domain)
            
            received_symbols.extend(data_symbols)
            
            if len(received_symbols) >= num_symbols:
                break
        
        received_symbols = np.array(received_symbols[:num_symbols])
        
        plt.figure(figsize=(12, 5))
        
        # Constelação recebida
        plt.subplot(1, 2, 1)
        plt.scatter(received_symbols.real, received_symbols.imag, 
                   alpha=0.5, s=20, label='Recebidos')
        plt.scatter(self.constellation.real, self.constellation.imag,
                   marker='x', s=200, c='red', linewidths=3, 
                   label='Constelação Ideal')
        plt.xlabel('I (In-phase)', fontsize=12)
        plt.ylabel('Q (Quadrature)', fontsize=12)
        plt.title(f'Constelação OFDM ({self.subcarrier_modulation})', 
                 fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # Histograma de EVM
        plt.subplot(1, 2, 2)
        
        # Calcula EVM para cada símbolo recebido
        evms = []
        for symbol in received_symbols:
            distances = np.abs(self.constellation - symbol)
            closest_idx = np.argmin(distances)
            ideal_symbol = self.constellation[closest_idx]
            evm = np.abs(symbol - ideal_symbol)
            evms.append(evm)
        
        evms = np.array(evms)
        
        plt.hist(evms, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('EVM (Error Vector Magnitude)', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Distribuição de EVM', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Estatísticas
        mean_evm = np.mean(evms)
        rms_evm = np.sqrt(np.mean(evms**2))
        plt.axvline(mean_evm, color='red', linestyle='--', 
                   label=f'Média: {mean_evm:.3f}')
        plt.axvline(rms_evm, color='orange', linestyle='--',
                   label=f'RMS: {rms_evm:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_spectrum(self, signal: np.ndarray):
        """
        Plota espectro do sinal recebido.
        
        Args:
            signal: Sinal recebido
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível")
            return
        
        # Downconvert
        baseband_signal = self.downconvert(signal)
        
        # FFT
        N = len(baseband_signal)
        fft_signal = np.fft.fftshift(np.fft.fft(baseband_signal))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/self.sample_rate))
        
        power_db = 20 * np.log10(np.abs(fft_signal) + 1e-10)
        
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(freqs/1e6, power_db, linewidth=0.8)
        plt.xlabel('Frequência (MHz)', fontsize=12)
        plt.ylabel('Potência (dB)', fontsize=12)
        plt.title('Espectro Recebido - OFDM', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Zoom
        plt.subplot(1, 2, 2)
        center_idx = N // 2
        zoom_range = self.num_subcarriers * 2
        start_idx = max(0, center_idx - zoom_range)
        end_idx = min(N, center_idx + zoom_range)
        
        plt.plot(freqs[start_idx:end_idx]/1e3, 
                power_db[start_idx:end_idx], linewidth=1.2)
        plt.xlabel('Frequência (kHz)', fontsize=12)
        plt.ylabel('Potência (dB)', fontsize=12)
        plt.title('Zoom - Subportadoras Individuais', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Marca posições das subportadoras
        subcarrier_spacing = self.sample_rate / self.num_subcarriers
        for i in range(-self.num_data_subcarriers//2, self.num_data_subcarriers//2):
            if i != 0:
                freq = i * subcarrier_spacing
                if start_idx <= center_idx + int(i * N / self.num_subcarriers) < end_idx:
                    plt.axvline(freq/1e3, color='red', alpha=0.2, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_evm_rms(self, signal: np.ndarray) -> float:
        """
        Calcula EVM RMS (Error Vector Magnitude) do sinal.
        
        Args:
            signal: Sinal recebido
            
        Returns:
            EVM RMS em porcentagem
        """
        baseband_signal = self.downconvert(signal)
        baseband_signal = self.apply_agc(baseband_signal)
        
        num_ofdm_symbols = len(baseband_signal) // self.ofdm_symbol_size
        
        all_errors = []
        
        for sym_idx in range(num_ofdm_symbols):
            start_idx = sym_idx * self.ofdm_symbol_size
            end_idx = start_idx + self.ofdm_symbol_size
            
            if end_idx > len(baseband_signal):
                break
            
            ofdm_symbol = baseband_signal[start_idx:end_idx]
            time_domain = self._remove_cyclic_prefix(ofdm_symbol)
            freq_domain = np.fft.fft(time_domain)
            data_symbols = self._extract_data_subcarriers(freq_domain)
            
            for symbol in data_symbols:
                distances = np.abs(self.constellation - symbol)
                closest_idx = np.argmin(distances)
                ideal_symbol = self.constellation[closest_idx]
                error = symbol - ideal_symbol
                all_errors.append(error)
        
        all_errors = np.array(all_errors)
        
        # EVM RMS
        error_power = np.mean(np.abs(all_errors)**2)
        signal_power = np.mean(np.abs(self.constellation)**2)
        evm_rms = np.sqrt(error_power / signal_power) * 100
        
        return evm_rms
    
    def __str__(self) -> str:
        return (f"OFDM Receiver:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Num Subcarriers: {self.num_subcarriers}\n"
                f"  Data Subcarriers: {self.num_data_subcarriers}\n"
                f"  CP Length: {self.cp_length}\n"
                f"  Subcarrier Modulation: {self.subcarrier_modulation}\n"
                f"  Bits/Subcarrier: {self.bits_per_subcarrier}\n"
                f"  Bits/OFDM Symbol: {self.bits_per_ofdm_symbol}\n"
                f"  OFDM Symbol Rate: {self.symbol_rate/1e3:.2f} ksym/s\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz")
