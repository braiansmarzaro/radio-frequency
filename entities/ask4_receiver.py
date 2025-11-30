"""
Receptor 4-ASK (4-level Amplitude Shift Keying).
Demodula sinais 4-ASK recuperando 2 bits por símbolo.
"""
import numpy as np
from .base_receiver import BaseReceiver


class ASK4Receiver(BaseReceiver):
    """
    Receptor 4-ASK (4-level Amplitude Shift Keying).
    
    Demodula sinais 4-ASK:
    - Recupera 2 bits por símbolo
    - Usa decisão de mínima distância
    - Apenas componente I é usada (Q=0)
    
    Taxa de recepção: 2 bits por símbolo
    """
    
    def __init__(self, sample_rate: float, symbol_rate: float, carrier_freq: float = 0):
        """
        Inicializa o receptor 4-ASK.
        
        Args:
            sample_rate: Taxa de amostragem em Hz
            symbol_rate: Taxa de símbolos em Hz
            carrier_freq: Frequência da portadora em Hz (0 para banda base)
        """
        super().__init__(sample_rate, carrier_freq)
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        
        # Gera níveis de amplitude de referência
        self.amplitude_levels = self._generate_amplitude_levels()
        
        # Limiares de decisão (pontos médios entre níveis)
        self.decision_thresholds = self._calculate_thresholds()
    
    def _generate_amplitude_levels(self) -> np.ndarray:
        """
        Gera os níveis de amplitude de referência (mesmos do transmissor).
        
        Returns:
            Array com 4 níveis de amplitude
        """
        levels = np.array([-3, -1, 1, 3], dtype=float)
        
        # Normaliza (mesma normalização do transmissor)
        avg_power = np.mean(levels**2)
        levels = levels / np.sqrt(avg_power)
        
        return levels
    
    def _calculate_thresholds(self) -> np.ndarray:
        """
        Calcula limiares de decisão entre níveis.
        
        Returns:
            Array com 3 limiares de decisão
        """
        # Limiares nos pontos médios entre níveis adjacentes
        thresholds = np.zeros(3)
        
        for i in range(3):
            thresholds[i] = (self.amplitude_levels[i] + self.amplitude_levels[i+1]) / 2
        
        return thresholds
    
    def demodulate(self, signal: np.ndarray, use_rrc: bool = False) -> np.ndarray:
        """
        Demodula sinal 4-ASK.
        
        Args:
            signal: Sinal I/Q recebido (complexo, mas apenas I é usado)
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
        
        # 4. Extrai componente real (I) - ASK não usa Q
        baseband_signal = np.real(baseband_signal)
        
        # 5. Amostragem nos instantes corretos
        symbols = self._sample_symbols(baseband_signal)
        
        # 6. Decisão: mapeia símbolos recebidos para bits
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
        
        # Filtragem (apenas componente real para ASK)
        signal_filtered = np.convolve(np.real(signal), rrc_filter, mode='same')
        
        return signal_filtered.astype(np.complex64)
    
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
            signal: Sinal em banda base (real)
            
        Returns:
            Array de símbolos (amplitudes)
        """
        # Amostragem simples
        num_symbols = len(signal) // self.samples_per_symbol
        symbols = signal[::self.samples_per_symbol][:num_symbols]
        
        return symbols
    
    def _symbol_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converte símbolos recebidos em bits usando limiares de decisão.
        
        Args:
            symbols: Símbolos (amplitudes) recebidos
            
        Returns:
            Array de bits
        """
        num_symbols = len(symbols)
        bits = np.zeros(num_symbols * 2, dtype=int)
        
        # Mapeamento Gray-coded reverso
        # Nível 0 (-3): 00
        # Nível 1 (-1): 01
        # Nível 2 (+1): 11
        # Nível 3 (+3): 10
        
        for i, symbol in enumerate(symbols):
            # Decisão baseada em limiares
            if symbol < self.decision_thresholds[0]:
                # Nível 0: bits = 00
                bits[i*2] = 0
                bits[i*2 + 1] = 0
            elif symbol < self.decision_thresholds[1]:
                # Nível 1: bits = 01
                bits[i*2] = 0
                bits[i*2 + 1] = 1
            elif symbol < self.decision_thresholds[2]:
                # Nível 2: bits = 11
                bits[i*2] = 1
                bits[i*2 + 1] = 1
            else:
                # Nível 3: bits = 10
                bits[i*2] = 1
                bits[i*2 + 1] = 0
        
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
        
        symbols = self._sample_symbols(np.real(baseband_signal))
        
        # Calcula LLRs
        num_symbols = len(symbols)
        llrs = np.zeros(num_symbols * 2, dtype=float)
        
        noise_var = self._estimate_noise_variance(symbols)
        
        for i, rx_symbol in enumerate(symbols):
            # Para cada posição de bit (0 ou 1)
            for bit_pos in range(2):
                # Níveis com bit=0 nesta posição
                if bit_pos == 0:  # Bit mais significativo
                    levels_0 = self.amplitude_levels[0:2]  # Níveis 0, 1
                    levels_1 = self.amplitude_levels[2:4]  # Níveis 2, 3
                else:  # Bit menos significativo
                    levels_0 = self.amplitude_levels[[0, 3]]  # Níveis 0, 3
                    levels_1 = self.amplitude_levels[[1, 2]]  # Níveis 1, 2
                
                # Calcula distâncias
                dist_0 = np.min((rx_symbol - levels_0)**2)
                dist_1 = np.min((rx_symbol - levels_1)**2)
                
                # LLR
                llrs[i*2 + bit_pos] = (dist_1 - dist_0) / (2 * noise_var)
        
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
            # Encontra nível mais próximo
            distances = np.abs(symbol - self.amplitude_levels)
            min_idx = np.argmin(distances)
            decided_symbols[i] = self.amplitude_levels[min_idx]
        
        # Erro = ruído + ISI
        error = symbols - decided_symbols
        noise_var = np.var(error)
        
        return max(noise_var, 1e-10)
    
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
    
    def plot_signal_diagram(self, signal: np.ndarray, use_rrc: bool = False, 
                           max_symbols: int = 200):
        """
        Plota diagrama de dispersão dos símbolos recebidos.
        
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
        
        symbols = self._sample_symbols(np.real(baseband_signal))
        symbols = symbols[:max_symbols]
        
        # Plota
        plt.figure(figsize=(12, 6))
        
        # Histograma dos símbolos
        plt.subplot(1, 2, 1)
        plt.hist(symbols, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Marca níveis ideais
        for level in self.amplitude_levels:
            plt.axvline(x=level, color='red', linestyle='--', linewidth=2, label='Níveis ideais')
        
        # Marca limiares
        for threshold in self.decision_thresholds:
            plt.axvline(x=threshold, color='green', linestyle=':', linewidth=1.5)
        
        plt.xlabel('Amplitude', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.title('Histograma de Símbolos Recebidos - 4-ASK', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Dispersão temporal
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(symbols)), symbols, s=10, alpha=0.6, c='blue')
        
        # Níveis ideais
        for level in self.amplitude_levels:
            plt.axhline(y=level, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        plt.xlabel('Índice do Símbolo', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title(f'Símbolos Recebidos ({len(symbols)} amostras)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_bit_rate(self) -> float:
        """
        Retorna a taxa de bits em bps.
        
        Returns:
            Taxa de bits (bps)
        """
        return self.symbol_rate * 2  # 4-ASK: 2 bits/símbolo
    
    def __str__(self) -> str:
        return (f"4-ASK Receiver:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Symbol Rate: {self.symbol_rate/1e3:.2f} kbaud\n"
                f"  Bit Rate: {self.get_bit_rate()/1e3:.2f} kbps\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz\n"
                f"  Samples/Symbol: {self.samples_per_symbol}\n"
                f"  Amplitude Levels: {self.amplitude_levels}\n"
                f"  Decision Thresholds: {self.decision_thresholds}")
