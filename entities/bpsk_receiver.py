"""
Receptor BPSK (Binary Phase Shift Keying).
Demodula sinais BPSK recuperando 1 bit por símbolo.
"""
import numpy as np
from .base_receiver import BaseReceiver
from typing import Optional


class BPSKReceiver(BaseReceiver):
    """
    Receptor BPSK (Binary Phase Shift Keying).
    
    Demodula sinais BPSK:
    - Recupera 1 bit por símbolo
    - Usa decisão de limiar em zero
    - Símbolo > 0 → bit 1, Símbolo < 0 → bit 0
    
    Taxa de recepção: 1 bit por símbolo
    """
    
    def __init__(self, sample_rate: float, symbol_rate: float, carrier_freq: float = 0):
        """
        Inicializa o receptor BPSK.
        
        Args:
            sample_rate: Taxa de amostragem em Hz
            symbol_rate: Taxa de símbolos em Hz
            carrier_freq: Frequência da portadora em Hz (0 para banda base)
        """
        super().__init__(sample_rate, carrier_freq)
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = int(sample_rate / symbol_rate)
        
    def demodulate(self, signal: np.ndarray, use_rrc: bool = False) -> np.ndarray:
        """
        Demodula sinal BPSK.
        
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
        
        # 4. Extrai componente real (BPSK é real)
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
        
        # Filtragem (apenas componente real para BPSK)
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
            Array de símbolos
        """
        # Amostragem simples
        num_symbols = len(signal) // self.samples_per_symbol
        symbols = signal[::self.samples_per_symbol][:num_symbols]
        
        return symbols
    
    def _symbol_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Converte símbolos recebidos em bits usando decisão de limiar.
        
        Args:
            symbols: Símbolos recebidos
            
        Returns:
            Array de bits
        """
        # Decisão simples: símbolo > 0 → bit 1, símbolo < 0 → bit 0
        # Corresponde ao mapeamento: bit 0 → -1, bit 1 → +1
        bits = (symbols > 0).astype(int)
        
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
        
        # Estima variância do ruído
        noise_var = self._estimate_noise_variance(symbols)
        
        # Para BPSK, LLR é simplesmente proporcional ao valor do símbolo
        # LLR = 2 * símbolo / noise_var
        # Valores positivos favorecem bit 1, negativos favorecem bit 0
        llrs = 2 * symbols / noise_var
        
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
        decided_symbols = np.where(symbols > 0, 1.0, -1.0)
        
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
                           max_symbols: int = 500):
        """
        Plota diagrama dos símbolos recebidos.
        
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
        
        # Cria figura com 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Histograma
        axes[0].hist(symbols, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(x=-1, color='red', linestyle='--', linewidth=2, label='Bit 0 (-1)')
        axes[0].axvline(x=1, color='green', linestyle='--', linewidth=2, label='Bit 1 (+1)')
        axes[0].axvline(x=0, color='black', linestyle=':', linewidth=1.5, label='Limiar')
        axes[0].set_xlabel('Amplitude', fontsize=12)
        axes[0].set_ylabel('Frequência', fontsize=12)
        axes[0].set_title('Histograma de Símbolos BPSK', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Subplot 2: Dispersão temporal
        axes[1].scatter(range(len(symbols)), symbols, s=10, alpha=0.6, c='blue')
        axes[1].axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Bit 0')
        axes[1].axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Bit 1')
        axes[1].axhline(y=0, color='black', linestyle=':', linewidth=1.5, label='Limiar')
        axes[1].set_xlabel('Índice do Símbolo', fontsize=12)
        axes[1].set_ylabel('Amplitude', fontsize=12)
        axes[1].set_title(f'Dispersão Temporal ({len(symbols)} símbolos)', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_eye_diagram(self, signal: np.ndarray, num_traces: int = 50, use_rrc: bool = False):
        """
        Plota diagrama de olho do sinal BPSK.
        
        Args:
            signal: Sinal recebido
            num_traces: Número de traços a plotar
            use_rrc: Se True, aplica filtro casado
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível")
            return
        
        # Processa sinal
        baseband_signal = self.downconvert(signal)
        
        if use_rrc:
            baseband_signal = self._apply_matched_filter(baseband_signal)
        
        baseband_signal = np.real(baseband_signal)
        
        # Prepara diagrama de olho
        samples_per_trace = self.samples_per_symbol * 2
        num_traces = min(num_traces, len(baseband_signal) // samples_per_trace)
        
        plt.figure(figsize=(10, 6))
        
        for i in range(num_traces):
            start = i * self.samples_per_symbol
            end = start + samples_per_trace
            
            if end <= len(baseband_signal):
                trace = baseband_signal[start:end]
                t = np.arange(len(trace)) / self.sample_rate * 1e6  # μs
                plt.plot(t, trace, 'b', alpha=0.3, linewidth=0.5)
        
        plt.axhline(y=0, color='k', linestyle=':', linewidth=1.5, label='Limiar de Decisão')
        plt.axhline(y=-1, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Bit 0')
        plt.axhline(y=1, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Bit 1')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Tempo (μs)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.title('Diagrama de Olho - BPSK', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def get_bit_rate(self) -> float:
        """
        Retorna a taxa de bits em bps.
        
        Returns:
            Taxa de bits (bps)
        """
        return self.symbol_rate  # BPSK: 1 bit/símbolo
    
    def __str__(self) -> str:
        return (f"BPSK Receiver:\n"
                f"  Sample Rate: {self.sample_rate/1e6:.2f} MS/s\n"
                f"  Symbol Rate: {self.symbol_rate/1e3:.2f} kbaud\n"
                f"  Bit Rate: {self.get_bit_rate()/1e3:.2f} kbps\n"
                f"  Carrier Freq: {self.carrier_freq/1e6:.2f} MHz\n"
                f"  Samples/Symbol: {self.samples_per_symbol}")
