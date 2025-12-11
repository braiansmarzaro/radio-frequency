#!/usr/bin/env python3
"""
Simulação de transmissão OFDM em canal AWGN
Permite configurar parâmetros e mede BER vs SNR
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from typing import Tuple


class OFDMTransmitter:
    """Transmissor OFDM configurável"""
    
    def __init__(self, 
                 bit_rate: float,           # Taxa de bits (bps)
                 bandwidth: float,          # Largura de banda (Hz)
                 max_delay: float,          # Atraso máximo do canal (s)
                 guard_time: float,         # Tempo de guarda (s)
                 symbol_time_multiplier: int = 4,  # Ts = multiplier * Tg
                 carrier_freq: float = None,  # Frequência da portadora (Hz), None = banda base
                 oversampling_factor: int = 8  # Fator de sobreamostragem para portadora
                 ):
        """
        Inicializa transmissor OFDM
        
        Args:
            bit_rate: Taxa de bits em bps
            bandwidth: Largura de banda disponível em Hz
            max_delay: Atraso máximo do canal em segundos
            guard_time: Tempo de guarda em segundos
            symbol_time_multiplier: Multiplicador para tempo de símbolo (Ts = m * Tg)
            carrier_freq: Frequência da portadora em Hz (None para banda base)
            oversampling_factor: Fator de sobreamostragem quando usar portadora
        """
        self.Rb = bit_rate
        self.BW = bandwidth
        self.max_delay = max_delay
        self.Tg = guard_time
        self.symbol_time_multiplier = symbol_time_multiplier
        self.carrier_freq = carrier_freq
        self.oversampling_factor = oversampling_factor
        
        # Calcula tempo de símbolo (múltiplo do tempo de guarda)
        self.Ts = self.symbol_time_multiplier * self.Tg
        
        # Tempo útil de símbolo (sem guarda)
        self.Tu = self.Ts - self.Tg
        
        # Número de subportadoras (baseado na largura de banda e tempo útil)
        self.N_subcarriers = int(self.BW * self.Tu)
        
        # Ajusta para potência de 2 (FFT eficiente)
        self.N_fft = 2 ** int(np.ceil(np.log2(self.N_subcarriers)))
        
        # Bits por símbolo OFDM (usando 16-QAM = 4 bits/subportadora)
        self.bits_per_symbol = 4  # 16-QAM
        self.bits_per_ofdm = self.N_fft * self.bits_per_symbol
        
        # Taxa efetiva de símbolos OFDM
        self.symbol_rate = 1 / self.Ts
        
        # Taxa de bits efetiva
        self.effective_bitrate = self.bits_per_ofdm * self.symbol_rate
        
        print("=" * 70)
        print("CONFIGURAÇÃO DO SISTEMA OFDM")
        print("=" * 70)
        print(f"Taxa de bits solicitada (Rb):      {self.Rb/1e6:.2f} Mbps")
        print(f"Largura de banda disponível:       {self.BW/1e6:.2f} MHz")
        print(f"Atraso máximo do canal:            {self.max_delay*1e6:.2f} µs")
        print(f"Tempo de guarda (Tg):              {self.Tg*1e6:.2f} µs")
        print(f"Tempo de símbolo (Ts):             {self.Ts*1e6:.2f} µs ({self.symbol_time_multiplier}×Tg)")
        print(f"Tempo útil (Tu):                   {self.Tu*1e6:.2f} µs")
        print(f"Número de subportadoras (FFT):     {self.N_fft}")
        print(f"Modulação:                         16-QAM ({self.bits_per_symbol} bits/subportadora)")
        print(f"Bits por símbolo OFDM:             {self.bits_per_ofdm}")
        print(f"Taxa de símbolos OFDM:             {self.symbol_rate/1e3:.2f} kSymbols/s")
        print(f"Taxa de bits efetiva:              {self.effective_bitrate/1e6:.2f} Mbps")
        print(f"Eficiência espectral:              {self.effective_bitrate/self.BW:.2f} bps/Hz")
        if self.carrier_freq is not None:
            print(f"Frequência da portadora:           {self.carrier_freq/1e6:.2f} MHz")
            print(f"Fator de sobreamostragem:          {self.oversampling_factor}×")
        else:
            print(f"Modo:                              Banda Base")
        print("=" * 70)
        
    def qam16_modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modulação 16-QAM com Gray coding
        4 bits -> 1 símbolo complexo
        Constelação: 4x4 grid normalizada
        """
        # Agrupa bits em grupos de 4
        bits = bits.reshape(-1, 4)
        
        # Mapeamento Gray-coded 16-QAM
        # Níveis I e Q: -3, -1, +1, +3
        gray_map = [
            0b0000, 0b0001, 0b0011, 0b0010,  # I=-3
            0b0100, 0b0101, 0b0111, 0b0110,  # I=-1
            0b1100, 0b1101, 0b1111, 0b1110,  # I=+1
            0b1000, 0b1001, 0b1011, 0b1010   # I=+3
        ]
        
        # Gera constelação
        levels = np.array([-3, -1, 1, 3])
        constellation = np.zeros(16, dtype=complex)
        
        idx = 0
        for i in range(4):
            for q in range(4):
                gray_idx = gray_map[idx]
                constellation[gray_idx] = levels[q] + 1j * levels[3-i]
                idx += 1
        
        # Normaliza potência média para 1
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        
        # Mapeia bits para símbolos
        symbols = np.zeros(len(bits), dtype=complex)
        for i in range(len(bits)):
            # Converte 4 bits para índice
            symbol_idx = 0
            for j in range(4):
                symbol_idx |= (bits[i, j] << (3 - j))
            symbols[i] = constellation[symbol_idx]
        
        return symbols
    
    def qam16_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Demodulação 16-QAM por mínima distância"""
        # Gera constelação de referência (mesma do modulador)
        gray_map = [
            0b0000, 0b0001, 0b0011, 0b0010,
            0b0100, 0b0101, 0b0111, 0b0110,
            0b1100, 0b1101, 0b1111, 0b1110,
            0b1000, 0b1001, 0b1011, 0b1010
        ]
        
        levels = np.array([-3, -1, 1, 3])
        constellation = np.zeros(16, dtype=complex)
        
        idx = 0
        for i in range(4):
            for q in range(4):
                gray_idx = gray_map[idx]
                constellation[gray_idx] = levels[q] + 1j * levels[3-i]
                idx += 1
        
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        
        # Demodula por mínima distância
        bits = np.zeros(len(symbols) * 4, dtype=int)
        
        for i, symbol in enumerate(symbols):
            # Encontra símbolo mais próximo
            distances = np.abs(constellation - symbol)
            closest_idx = np.argmin(distances)
            
            # Converte índice para 4 bits
            for j in range(4):
                bits[i*4 + j] = (closest_idx >> (3 - j)) & 1
        
        return bits
    
    def add_cyclic_prefix(self, ofdm_symbol: np.ndarray) -> np.ndarray:
        """Adiciona prefixo cíclico"""
        cp_length = int(self.Tg / self.Tu * self.N_fft)
        cp = ofdm_symbol[-cp_length:]
        return np.concatenate([cp, ofdm_symbol])
    
    def remove_cyclic_prefix(self, ofdm_symbol: np.ndarray) -> np.ndarray:
        """Remove prefixo cíclico"""
        cp_length = int(self.Tg / self.Tu * self.N_fft)
        return ofdm_symbol[cp_length:]
    
    def upconvert_to_carrier(self, baseband_signal: np.ndarray) -> np.ndarray:
        """
        Converte sinal de banda base para portadora
        
        Args:
            baseband_signal: Sinal complexo em banda base
            
        Returns:
            Sinal real modulado em portadora
        """
        if self.carrier_freq is None:
            # Retorna sinal em banda base (apenas parte real)
            return np.real(baseband_signal)
        
        # Reamostragem do sinal
        num_samples_original = len(baseband_signal)
        num_samples_upsampled = num_samples_original * self.oversampling_factor
        
        # Interpola o sinal para maior taxa de amostragem
        baseband_upsampled = np.zeros(num_samples_upsampled, dtype=complex)
        baseband_upsampled[::self.oversampling_factor] = baseband_signal
        
        # Filtro de interpolação (passa-baixas)
        b = scipy_signal.firwin(64, 1.0/self.oversampling_factor)
        baseband_upsampled = scipy_signal.lfilter(b, 1, baseband_upsampled) * self.oversampling_factor
        
        # Gera vetor de tempo
        sample_rate_baseband = self.N_fft / self.Tu  # Taxa de amostragem original
        sample_rate_upsampled = sample_rate_baseband * self.oversampling_factor
        t = np.arange(num_samples_upsampled) / sample_rate_upsampled
        
        # Modulação em portadora
        # s(t) = I(t) * cos(2πfc*t) - Q(t) * sin(2πfc*t)
        I = np.real(baseband_upsampled)
        Q = np.imag(baseband_upsampled)
        
        carrier_signal = I * np.cos(2 * np.pi * self.carrier_freq * t) - Q * np.sin(2 * np.pi * self.carrier_freq * t)
        
        return carrier_signal
    
    def downconvert_from_carrier(self, carrier_signal: np.ndarray, original_length: int) -> np.ndarray:
        """
        Converte sinal de portadora para banda base
        
        Args:
            carrier_signal: Sinal real modulado em portadora
            original_length: Comprimento do sinal original em banda base
            
        Returns:
            Sinal complexo em banda base
        """
        if self.carrier_freq is None:
            # Sinal já está em banda base
            return carrier_signal.astype(complex)
        
        # Gera vetor de tempo
        sample_rate_baseband = self.N_fft / self.Tu
        sample_rate_upsampled = sample_rate_baseband * self.oversampling_factor
        t = np.arange(len(carrier_signal)) / sample_rate_upsampled
        
        # Demodulação
        # I(t) = s(t) * cos(2πfc*t) * 2
        # Q(t) = -s(t) * sin(2πfc*t) * 2
        I_demod = carrier_signal * np.cos(2 * np.pi * self.carrier_freq * t) * 2
        Q_demod = -carrier_signal * np.sin(2 * np.pi * self.carrier_freq * t) * 2
        
        # Filtro passa-baixas para remover componentes de alta frequência
        cutoff = 1.0 / self.oversampling_factor
        b = scipy_signal.firwin(128, cutoff)
        
        I_filtered = scipy_signal.lfilter(b, 1, I_demod)
        Q_filtered = scipy_signal.lfilter(b, 1, Q_demod)
        
        # Reconstrói sinal complexo
        baseband_upsampled = I_filtered + 1j * Q_filtered
        
        # Decimação (reamostragem para taxa original)
        baseband_signal = baseband_upsampled[::self.oversampling_factor]
        
        # Ajusta comprimento
        return baseband_signal[:original_length]
    
    def transmit_frame(self, num_symbols: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transmite um frame de símbolos OFDM
        
        Args:
            num_symbols: Número de símbolos OFDM no frame
            
        Returns:
            tx_bits: Bits transmitidos
            tx_signal: Sinal transmitido (banda base ou com portadora)
        """
        # Gera bits aleatórios
        total_bits = num_symbols * self.bits_per_ofdm
        tx_bits = np.random.randint(0, 2, total_bits)
        
        # Sinal transmitido completo em banda base
        tx_signal_baseband = np.array([], dtype=complex)
        
        # Processa cada símbolo OFDM
        for i in range(num_symbols):
            # Extrai bits deste símbolo
            start_idx = i * self.bits_per_ofdm
            end_idx = start_idx + self.bits_per_ofdm
            symbol_bits = tx_bits[start_idx:end_idx]
            
            # Modula em 16-QAM
            qam16_symbols = self.qam16_modulate(symbol_bits)
            
            # IFFT (converte frequência -> tempo)
            ofdm_time = np.fft.ifft(qam16_symbols, self.N_fft)
            
            # Adiciona prefixo cíclico
            ofdm_with_cp = self.add_cyclic_prefix(ofdm_time)
            
            # Concatena ao sinal
            tx_signal_baseband = np.concatenate([tx_signal_baseband, ofdm_with_cp])
        
        # Converte para portadora se necessário
        tx_signal = self.upconvert_to_carrier(tx_signal_baseband)
        
        return tx_bits, tx_signal
    
    def receive_frame(self, rx_signal: np.ndarray, num_symbols: int = 12) -> np.ndarray:
        """
        Recebe e demodula um frame OFDM
        
        Args:
            rx_signal: Sinal recebido (banda base ou com portadora)
            num_symbols: Número de símbolos OFDM esperados
            
        Returns:
            rx_bits: Bits recebidos
        """
        cp_length = int(self.Tg / self.Tu * self.N_fft)
        symbol_length = self.N_fft + cp_length
        
        # Converte de portadora para banda base se necessário
        expected_baseband_length = num_symbols * symbol_length
        rx_signal_baseband = self.downconvert_from_carrier(rx_signal, expected_baseband_length)
        
        rx_bits = np.array([], dtype=int)
        
        # Processa cada símbolo OFDM
        for i in range(num_symbols):
            # Extrai símbolo do sinal
            start_idx = i * symbol_length
            end_idx = start_idx + symbol_length
            
            if end_idx > len(rx_signal_baseband):
                break
                
            ofdm_with_cp = rx_signal_baseband[start_idx:end_idx]
            
            # Remove prefixo cíclico
            ofdm_time = self.remove_cyclic_prefix(ofdm_with_cp)
            
            # FFT (converte tempo -> frequência)
            qam16_symbols = np.fft.fft(ofdm_time, self.N_fft)
            
            # Demodula 16-QAM
            symbol_bits = self.qam16_demodulate(qam16_symbols)
            
            # Concatena bits
            rx_bits = np.concatenate([rx_bits, symbol_bits])
        
        return rx_bits


def awgn_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Adiciona ruído AWGN ao sinal
    
    Args:
        signal: Sinal de entrada
        snr_db: SNR em dB
        
    Returns:
        Sinal com ruído
    """
    # Potência do sinal
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # Potência do ruído baseada no SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Gera ruído complexo AWGN
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    
    return signal + noise


def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Calcula taxa de erro de bit (BER)"""
    # Garante mesmo tamanho
    min_len = min(len(tx_bits), len(rx_bits))
    tx_bits = tx_bits[:min_len]
    rx_bits = rx_bits[:min_len]
    
    # Conta erros
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)
    
    return ber


def simulate_ber_vs_snr(
    ofdm: OFDMTransmitter,
    snr_range: np.ndarray,
    num_symbols: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula BER vs SNR
    
    Args:
        ofdm: Transmissor OFDM configurado
        snr_range: Array de valores de SNR em dB
        num_symbols: Número de símbolos por frame
        
    Returns:
        snr_values: Valores de SNR
        ber_values: Valores de BER correspondentes
    """
    ber_values = []
    
    print("\nSIMULAÇÃO BER vs SNR")
    print("=" * 70)
    
    for snr_db in snr_range:
        # Transmite frame
        tx_bits, tx_signal = ofdm.transmit_frame(num_symbols)
        
        # Canal AWGN
        rx_signal = awgn_channel(tx_signal, snr_db)
        
        # Recebe frame
        rx_bits = ofdm.receive_frame(rx_signal, num_symbols)
        
        # Calcula BER
        ber = calculate_ber(tx_bits, rx_bits)
        ber_values.append(ber)
        
        print(f"SNR: {snr_db:5.1f} dB | BER: {ber:.6e} | Erros: {int(ber * len(tx_bits))}/{len(tx_bits)}")
    
    print("=" * 70)
    
    return snr_range, np.array(ber_values)


def plot_ber_vs_snr(snr_values: np.ndarray, ber_values: np.ndarray):
    """Plota gráfico BER vs SNR"""
    plt.figure(figsize=(12, 7))
    
    # Gráfico em escala logarítmica
    plt.semilogy(snr_values, ber_values, 'b-o', linewidth=2, markersize=8, label='OFDM QPSK')
    
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('BER (Bit Error Rate)', fontsize=12, fontweight='bold')
    plt.title('Desempenho OFDM em Canal AWGN', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    
    # Formata eixos
    plt.xlim([snr_values[0] - 1, snr_values[-1] + 1])
    
    # Adiciona grade fina
    plt.grid(True, which='minor', alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('ofdm_ber_vs_snr.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico salvo em: ofdm_ber_vs_snr.png")
    #plt.show()


def main():
    """Função principal"""
    
    # ============================================================================
    # CONFIGURAÇÃO DOS PARÂMETROS
    # ============================================================================
    
    # Taxa de bits desejada (Mbps)
    bit_rate = 10e6  # 10 Mbps
    
    # Largura de banda disponível (MHz)
    bandwidth = 200e6  # 200 MHz
    
    # Atraso máximo do canal (µs)
    max_delay = 200e-9  # 0.2 µs
    
    # Tempo de guarda
    guard_time = 2*max_delay  # 0.4 µs
    
    # Tempo de símbolo como múltiplo do tempo de guarda
    symbol_time_multiplier = 7  # Ts = 7 × Tg = 2.8 µs
    
    # Frequência da portadora (None para banda base)
    # carrier_freq = None  # Banda base
    carrier_freq = 2.4e9  # 2.4 GHz (WiFi)
    
    # Fator de sobreamostragem para portadora
    oversampling_factor = 8
    
    # Número de símbolos OFDM por frame
    num_symbols = 12
    
    # Range de SNR para teste
    snr_range = np.arange(0, 31, 2)  # 0 a 30 dB, passo de 2 dB
    
    # ============================================================================
    # CRIA TRANSMISSOR OFDM
    # ============================================================================
    
    ofdm = OFDMTransmitter(
        bit_rate=bit_rate,
        bandwidth=bandwidth,
        max_delay=max_delay,
        guard_time=guard_time,
        symbol_time_multiplier=symbol_time_multiplier,
        carrier_freq=carrier_freq,
        oversampling_factor=oversampling_factor
    )
    
    # ============================================================================
    # SIMULA BER vs SNR
    # ============================================================================
    
    snr_values, ber_values = simulate_ber_vs_snr(ofdm, snr_range, num_symbols)
    
    # ============================================================================
    # EXIBE MÉTRICAS
    # ============================================================================
    
    print("\nMÉTRICAS DE TRANSMISSÃO")
    print("=" * 70)
    print(f"Total de bits transmitidos por frame:  {num_symbols * ofdm.bits_per_ofdm}")
    print(f"Duração do frame:                      {num_symbols * ofdm.Ts * 1e6:.2f} µs")
    print(f"Taxa de transmissão efetiva:           {ofdm.effective_bitrate/1e6:.2f} Mbps")
    print(f"Overhead do prefixo cíclico:           {(ofdm.Tg/ofdm.Ts)*100:.1f}%")
    print("=" * 70)
    
    # ============================================================================
    # PLOTA RESULTADOS
    # ============================================================================
    
    plot_ber_vs_snr(snr_values, ber_values)


if __name__ == "__main__":
    main()
