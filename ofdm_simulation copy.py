#!/usr/bin/env python3
"""
Simulação de transmissão OFDM em canal AWGN
Permite configurar parâmetros e mede BER vs SNR
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class OFDMTransmitter:
    """Transmissor OFDM configurável"""
    
    def __init__(self, 
                 bit_rate: float,           # Taxa de bits (bps)
                 bandwidth: float,          # Largura de banda (Hz)
                 max_delay: float,          # Atraso máximo do canal (s)
                 guard_time: float,         # Tempo de guarda (s)
                 symbol_time_multiplier: int = 4  # Ts = multiplier * Tg
                 ):
        """
        Inicializa transmissor OFDM
        
        Args:
            bit_rate: Taxa de bits em bps
            bandwidth: Largura de banda disponível em Hz
            max_delay: Atraso máximo do canal em segundos
            guard_time: Tempo de guarda em segundos
            symbol_time_multiplier: Multiplicador para tempo de símbolo (Ts = m * Tg)
        """
        self.Rb = bit_rate
        self.BW = bandwidth
        self.max_delay = max_delay
        self.Tg = guard_time
        self.symbol_time_multiplier = symbol_time_multiplier
        
        # Calcula tempo de símbolo (múltiplo do tempo de guarda)
        self.Ts = self.symbol_time_multiplier * self.Tg
        
        # Tempo útil de símbolo (sem guarda)
        self.Tu = self.Ts - self.Tg
        
        # Número de subportadoras (baseado na largura de banda e tempo útil)
        self.N_subcarriers = int(self.BW * self.Tu)
        
        # Ajusta para potência de 2 (FFT eficiente)
        self.N_fft = 2 ** int(np.ceil(np.log2(self.N_subcarriers)))
        
        # Bits por símbolo OFDM (usando QPSK = 2 bits/subportadora)
        self.bits_per_symbol = 2  # QPSK
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
        print(f"Modulação:                         QPSK ({self.bits_per_symbol} bits/subportadora)")
        print(f"Bits por símbolo OFDM:             {self.bits_per_ofdm}")
        print(f"Taxa de símbolos OFDM:             {self.symbol_rate/1e3:.2f} kSymbols/s")
        print(f"Taxa de bits efetiva:              {self.effective_bitrate/1e6:.2f} Mbps")
        print(f"Eficiência espectral:              {self.effective_bitrate/self.BW:.2f} bps/Hz")
        print("=" * 70)
        
    def qpsk_modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Modulação QPSK
        00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
        """
        # Agrupa bits em pares
        bits = bits.reshape(-1, 2)
        
        # Mapeia para símbolos QPSK
        symbols = np.zeros(len(bits), dtype=complex)
        symbols = (2*bits[:, 0] - 1) + 1j*(2*bits[:, 1] - 1)
        
        # Normalização
        symbols = symbols / np.sqrt(2)
        
        return symbols
    
    def qpsk_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Demodulação QPSK"""
        bits = np.zeros(len(symbols) * 2, dtype=int)
        
        # Recupera bits da parte real
        bits[0::2] = (np.real(symbols) > 0).astype(int)
        
        # Recupera bits da parte imaginária
        bits[1::2] = (np.imag(symbols) > 0).astype(int)
        
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
    
    def transmit_frame(self, num_symbols: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transmite um frame de símbolos OFDM
        
        Args:
            num_symbols: Número de símbolos OFDM no frame
            
        Returns:
            tx_bits: Bits transmitidos
            tx_signal: Sinal transmitido (domínio do tempo)
        """
        # Gera bits aleatórios
        total_bits = num_symbols * self.bits_per_ofdm
        tx_bits = np.random.randint(0, 2, total_bits)
        
        # Sinal transmitido completo
        tx_signal = np.array([], dtype=complex)
        
        # Processa cada símbolo OFDM
        for i in range(num_symbols):
            # Extrai bits deste símbolo
            start_idx = i * self.bits_per_ofdm
            end_idx = start_idx + self.bits_per_ofdm
            symbol_bits = tx_bits[start_idx:end_idx]
            
            # Modula em QPSK
            qpsk_symbols = self.qpsk_modulate(symbol_bits)
            
            # IFFT (converte frequência -> tempo)
            ofdm_time = np.fft.ifft(qpsk_symbols, self.N_fft)
            
            # Adiciona prefixo cíclico
            ofdm_with_cp = self.add_cyclic_prefix(ofdm_time)
            
            # Concatena ao sinal
            tx_signal = np.concatenate([tx_signal, ofdm_with_cp])
        
        return tx_bits, tx_signal
    
    def receive_frame(self, rx_signal: np.ndarray, num_symbols: int = 12) -> np.ndarray:
        """
        Recebe e demodula um frame OFDM
        
        Args:
            rx_signal: Sinal recebido
            num_symbols: Número de símbolos OFDM esperados
            
        Returns:
            rx_bits: Bits recebidos
        """
        cp_length = int(self.Tg / self.Tu * self.N_fft)
        symbol_length = self.N_fft + cp_length
        
        rx_bits = np.array([], dtype=int)
        
        # Processa cada símbolo OFDM
        for i in range(num_symbols):
            # Extrai símbolo do sinal
            start_idx = i * symbol_length
            end_idx = start_idx + symbol_length
            
            if end_idx > len(rx_signal):
                break
                
            ofdm_with_cp = rx_signal[start_idx:end_idx]
            
            # Remove prefixo cíclico
            ofdm_time = self.remove_cyclic_prefix(ofdm_with_cp)
            
            # FFT (converte tempo -> frequência)
            qpsk_symbols = np.fft.fft(ofdm_time, self.N_fft)
            
            # Demodula QPSK
            symbol_bits = self.qpsk_demodulate(qpsk_symbols)
            
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
    bandwidth = 20e6  # 20 MHz
    
    # Atraso máximo do canal (µs)
    max_delay = 2*200e-9  # 0.4 µs
    
    # Tempo de guarda
    guard_time = 400e-9  # 0.4 µs
    
    # Tempo de símbolo como múltiplo do tempo de guarda
    symbol_time_multiplier = 7  # Ts = 7 × Tg = 2.8 µs
    
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
        symbol_time_multiplier=symbol_time_multiplier
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
