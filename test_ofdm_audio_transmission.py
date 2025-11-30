"""
Teste de transmissão de áudio usando OFDM.
Áudio de 5 segundos, mono, 16 kHz, 16 bits por amostra.
"""
import numpy as np
import matplotlib.pyplot as plt
from entities import OFDMTransmitter, OFDMReceiver, AWGNChannel, Transmission


def generate_test_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Gera áudio de teste com múltiplas frequências (simulando voz/música).
    
    Args:
        duration: Duração em segundos
        sample_rate: Taxa de amostragem em Hz
        
    Returns:
        Array de amostras de áudio (valores int16)
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Mistura de frequências simulando áudio complexo
    # Fundamental e harmônicos (simulando voz humana)
    f0 = 200  # Frequência fundamental (Hz)
    audio = np.zeros(num_samples)
    
    # Adiciona fundamental e harmônicos com amplitude decrescente
    audio += 0.3 * np.sin(2 * np.pi * f0 * t)  # Fundamental
    audio += 0.2 * np.sin(2 * np.pi * 2 * f0 * t)  # 2º harmônico
    audio += 0.15 * np.sin(2 * np.pi * 3 * f0 * t)  # 3º harmônico
    audio += 0.1 * np.sin(2 * np.pi * 4 * f0 * t)  # 4º harmônico
    
    # Adiciona componentes de frequência mais alta (consoantes)
    audio += 0.15 * np.sin(2 * np.pi * 1000 * t) * np.sin(2 * np.pi * 5 * t)
    audio += 0.1 * np.sin(2 * np.pi * 2000 * t) * np.sin(2 * np.pi * 3 * t)
    
    # Adiciona modulação de amplitude (envelope)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio = audio * envelope
    
    # Normaliza e converte para int16
    audio = audio / np.max(np.abs(audio))  # Normaliza para [-1, 1]
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16


def calculate_snr_audio(original: np.ndarray, received: np.ndarray) -> float:
    """
    Calcula SNR do áudio recebido.
    
    Args:
        original: Áudio original
        received: Áudio recebido
        
    Returns:
        SNR em dB
    """
    # Garante mesmo tamanho
    min_len = min(len(original), len(received))
    original = original[:min_len]
    received = received[:min_len]
    
    # Calcula potência do sinal e do ruído
    signal_power = np.mean(original.astype(float)**2)
    noise = original.astype(float) - received.astype(float)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def calculate_thd(audio: np.ndarray, sample_rate: int, fundamental_freq: float = 200) -> float:
    """
    Calcula THD (Total Harmonic Distortion).
    
    Args:
        audio: Sinal de áudio
        sample_rate: Taxa de amostragem
        fundamental_freq: Frequência fundamental
        
    Returns:
        THD em porcentagem
    """
    # FFT
    fft_audio = np.fft.rfft(audio.astype(float))
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    
    # Encontra potência da fundamental
    fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
    fund_power = np.abs(fft_audio[fund_idx])**2
    
    # Soma potência dos harmônicos (2º ao 5º)
    harmonic_power = 0
    for n in range(2, 6):
        harm_freq = n * fundamental_freq
        harm_idx = np.argmin(np.abs(freqs - harm_freq))
        harmonic_power += np.abs(fft_audio[harm_idx])**2
    
    if fund_power == 0:
        return 0
    
    thd = np.sqrt(harmonic_power / fund_power) * 100
    return thd


def main():
    print("=" * 70)
    print("TESTE DE TRANSMISSÃO DE ÁUDIO COM OFDM")
    print("=" * 70)
    
    # Parâmetros do áudio
    audio_duration = 5.0  # segundos
    audio_sample_rate = 16000  # Hz
    bits_per_sample = 16
    
    print(f"\nParâmetros do Áudio:")
    print(f"  Duração: {audio_duration} s")
    print(f"  Taxa de Amostragem: {audio_sample_rate} Hz")
    print(f"  Resolução: {bits_per_sample} bits/amostra")
    print(f"  Canais: Mono")
    
    # Gera áudio de teste
    print("\nGerando áudio de teste...")
    audio_samples = generate_test_audio(audio_duration, audio_sample_rate)
    
    audio_size_bytes = len(audio_samples) * 2  # 2 bytes por amostra (int16)
    audio_size_bits = audio_size_bytes * 8
    
    print(f"  Amostras: {len(audio_samples)}")
    print(f"  Tamanho: {audio_size_bytes} bytes ({audio_size_bits} bits)")
    
    # Parâmetros OFDM
    # Para transmitir em tempo razoável, usamos alta taxa de símbolos
    ofdm_sample_rate = 1e6  # 1 MS/s
    num_subcarriers = 128
    cp_length = 32
    subcarrier_mod = 'QPSK'  # 2 bits/subportadora
    carrier_freq = 2.4e9  # 2.4 GHz (WiFi-like)
    
    print(f"\nParâmetros OFDM:")
    print(f"  Taxa de Amostragem: {ofdm_sample_rate/1e6:.1f} MS/s")
    print(f"  Número de Subportadoras: {num_subcarriers}")
    print(f"  Prefixo Cíclico: {cp_length}")
    print(f"  Modulação por Subportadora: {subcarrier_mod}")
    print(f"  Frequência da Portadora: {carrier_freq/1e9:.1f} GHz")
    
    # Cria transmissor e receptor OFDM
    transmitter = OFDMTransmitter(
        sample_rate=ofdm_sample_rate,
        num_subcarriers=num_subcarriers,
        cp_length=cp_length,
        subcarrier_modulation=subcarrier_mod,
        carrier_freq=carrier_freq
    )
    
    receiver = OFDMReceiver(
        sample_rate=ofdm_sample_rate,
        num_subcarriers=num_subcarriers,
        cp_length=cp_length,
        subcarrier_modulation=subcarrier_mod,
        carrier_freq=carrier_freq
    )
    
    print(f"\n{transmitter}")
    
    # Calcula tempo de transmissão esperado
    bit_rate = transmitter.get_bit_rate()
    expected_transmission_time = audio_size_bits / bit_rate
    
    print(f"\nTempo de Transmissão Esperado: {expected_transmission_time*1000:.2f} ms")
    print(f"Taxa de Bits: {bit_rate/1e6:.3f} Mbps")
    print(f"Eficiência Espectral: {transmitter.get_spectral_efficiency():.2f} bits/s/Hz")
    
    # Canal AWGN
    channel_snr = 20  # dB
    channel = AWGNChannel(snr_db=channel_snr)
    
    # Cria sistema de transmissão
    system = Transmission(transmitter, channel, receiver)
    
    # Converte áudio para bytes
    audio_bytes = audio_samples.tobytes()
    
    print(f"\nIniciando transmissão (SNR do canal: {channel_snr} dB)...")
    
    # Transmite
    rx_bytes, _ = system.transmit_bytes(audio_bytes)
    
    # Métricas
    metrics = system.get_metrics()
    
    print(f"\n{system}")
    
    # Converte bytes recebidos de volta para amostras de áudio
    # Garante que temos número par de bytes
    if len(rx_bytes) % 2 != 0:
        rx_bytes = rx_bytes[:-1]
    
    # Limita ao tamanho original
    rx_bytes = rx_bytes[:audio_size_bytes]
    
    # Converte para int16
    rx_audio = np.frombuffer(rx_bytes, dtype=np.int16)
    
    # Garante mesmo tamanho
    min_len = min(len(audio_samples), len(rx_audio))
    audio_samples = audio_samples[:min_len]
    rx_audio = rx_audio[:min_len]
    
    # Calcula SNR do áudio
    audio_snr = calculate_snr_audio(audio_samples, rx_audio)
    
    # Calcula THD
    thd_original = calculate_thd(audio_samples, audio_sample_rate)
    thd_received = calculate_thd(rx_audio, audio_sample_rate)
    
    print(f"\nQualidade do Áudio:")
    print(f"  SNR do Áudio: {audio_snr:.2f} dB")
    print(f"  THD Original: {thd_original:.2f}%")
    print(f"  THD Recebido: {thd_received:.2f}%")
    
    # Calcula EVM
    evm_rms = receiver.calculate_evm_rms(system.rx_signal)
    print(f"  EVM RMS: {evm_rms:.2f}%")
    
    # Visualização
    print("\nGerando visualizações...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Forma de onda - Original vs Recebido
    ax1 = plt.subplot(3, 3, 1)
    time_audio = np.arange(len(audio_samples)) / audio_sample_rate
    display_samples = min(1000, len(audio_samples))  # Mostra primeiros 1000 amostras
    ax1.plot(time_audio[:display_samples]*1000, audio_samples[:display_samples], 
             label='Original', alpha=0.7, linewidth=1)
    ax1.plot(time_audio[:display_samples]*1000, rx_audio[:display_samples], 
             label='Recebido', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Tempo (ms)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('Forma de Onda (primeiros 62.5 ms)', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Espectro de Frequência - Original
    ax2 = plt.subplot(3, 3, 2)
    fft_original = np.fft.rfft(audio_samples.astype(float))
    freqs = np.fft.rfftfreq(len(audio_samples), 1/audio_sample_rate)
    power_db_original = 20 * np.log10(np.abs(fft_original) + 1e-10)
    ax2.plot(freqs/1000, power_db_original, linewidth=0.8)
    ax2.set_xlabel('Frequência (kHz)', fontsize=10)
    ax2.set_ylabel('Magnitude (dB)', fontsize=10)
    ax2.set_title('Espectro - Áudio Original', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, audio_sample_rate/2000])
    
    # 3. Espectro de Frequência - Recebido
    ax3 = plt.subplot(3, 3, 3)
    fft_received = np.fft.rfft(rx_audio.astype(float))
    power_db_received = 20 * np.log10(np.abs(fft_received) + 1e-10)
    ax3.plot(freqs/1000, power_db_received, linewidth=0.8, color='orange')
    ax3.set_xlabel('Frequência (kHz)', fontsize=10)
    ax3.set_ylabel('Magnitude (dB)', fontsize=10)
    ax3.set_title('Espectro - Áudio Recebido', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, audio_sample_rate/2000])
    
    # 4. Diferença (Erro)
    ax4 = plt.subplot(3, 3, 4)
    error_signal = audio_samples.astype(float) - rx_audio.astype(float)
    ax4.plot(time_audio[:display_samples]*1000, error_signal[:display_samples], 
             color='red', linewidth=0.8)
    ax4.set_xlabel('Tempo (ms)', fontsize=10)
    ax4.set_ylabel('Erro', fontsize=10)
    ax4.set_title('Sinal de Erro', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Histograma de Erro
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(error_signal, bins=100, alpha=0.7, edgecolor='black', color='red')
    ax5.set_xlabel('Amplitude do Erro', fontsize=10)
    ax5.set_ylabel('Frequência', fontsize=10)
    ax5.set_title('Distribuição do Erro', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Constelação OFDM
    ax6 = plt.subplot(3, 3, 6)
    # Extrai alguns símbolos para plotar
    baseband = receiver.downconvert(system.rx_signal)
    baseband = receiver.apply_agc(baseband)
    num_ofdm_syms = min(100, len(baseband) // receiver.ofdm_symbol_size)
    
    symbols_to_plot = []
    for i in range(num_ofdm_syms):
        start = i * receiver.ofdm_symbol_size
        end = start + receiver.ofdm_symbol_size
        if end > len(baseband):
            break
        ofdm_sym = baseband[start:end]
        time_dom = receiver._remove_cyclic_prefix(ofdm_sym)
        freq_dom = np.fft.fft(time_dom)
        data_syms = receiver._extract_data_subcarriers(freq_dom)
        symbols_to_plot.extend(data_syms[:50])  # 50 símbolos por símbolo OFDM
    
    symbols_to_plot = np.array(symbols_to_plot)
    ax6.scatter(symbols_to_plot.real, symbols_to_plot.imag, 
               alpha=0.3, s=10, label='Recebidos')
    ax6.scatter(receiver.constellation.real, receiver.constellation.imag,
               marker='x', s=200, c='red', linewidths=3, label='Ideal')
    ax6.set_xlabel('I', fontsize=10)
    ax6.set_ylabel('Q', fontsize=10)
    ax6.set_title(f'Constelação OFDM ({subcarrier_mod})', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    
    # 7. Espectro OFDM
    ax7 = plt.subplot(3, 3, 7)
    N = len(system.tx_signal)
    fft_ofdm = np.fft.fftshift(np.fft.fft(system.tx_signal))
    freqs_ofdm = np.fft.fftshift(np.fft.fftfreq(N, 1/ofdm_sample_rate))
    power_ofdm = 20 * np.log10(np.abs(fft_ofdm) + 1e-10)
    ax7.plot(freqs_ofdm/1e6, power_ofdm, linewidth=0.6)
    ax7.set_xlabel('Frequência (MHz)', fontsize=10)
    ax7.set_ylabel('Potência (dB)', fontsize=10)
    ax7.set_title('Espectro do Sinal OFDM Transmitido', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. BER vs Tempo
    ax8 = plt.subplot(3, 3, 8)
    # Calcula BER em blocos
    block_size = len(audio_samples) // 20  # 20 blocos
    ber_blocks = []
    time_blocks = []
    
    for i in range(20):
        start = i * block_size
        end = min((i + 1) * block_size, len(audio_samples))
        if end > len(rx_audio):
            break
        
        orig_block = audio_samples[start:end]
        rx_block = rx_audio[start:end]
        
        # Converte para bits
        orig_bits = np.unpackbits(orig_block.view(np.uint8))
        rx_bits = np.unpackbits(rx_block.view(np.uint8))
        
        # BER
        errors = np.sum(orig_bits != rx_bits)
        ber = errors / len(orig_bits) if len(orig_bits) > 0 else 0
        
        ber_blocks.append(ber)
        time_blocks.append((start + end) / 2 / audio_sample_rate)
    
    ax8.plot(time_blocks, ber_blocks, marker='o', linewidth=2, markersize=6)
    ax8.set_xlabel('Tempo (s)', fontsize=10)
    ax8.set_ylabel('BER', fontsize=10)
    ax8.set_title('Taxa de Erro de Bits ao Longo do Tempo', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    # Só usa escala log se houver valores positivos
    if any(ber > 0 for ber in ber_blocks):
        ax8.set_yscale('log')
    
    # 9. Métricas Resumidas
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    metrics_text = (
        f"MÉTRICAS DO SISTEMA\n"
        f"{'-'*35}\n\n"
        f"Áudio:\n"
        f"  SNR: {audio_snr:.2f} dB\n"
        f"  THD Original: {thd_original:.2f}%\n"
        f"  THD Recebido: {thd_received:.2f}%\n\n"
        f"Transmissão OFDM:\n"
        f"  BER: {metrics['ber']:.2e}\n"
        + (f"  SNR Canal: {metrics['snr_db']:.2f} dB\n" if metrics['snr_db'] is not None else "  SNR Canal: N/A\n") +
        f"  EVM RMS: {evm_rms:.2f}%\n"
        f"  Tempo: {metrics['transmission_time']*1000:.2f} ms\n"
        f"  Banda: {metrics['bandwidth_used']/1e6:.2f} MHz\n"
        f"  Taxa: {bit_rate/1e6:.3f} Mbps\n"
        + (f"  Throughput: {metrics['throughput']/1e6:.2f} Mbps" if metrics['throughput'] is not None else "  Throughput: N/A")
    )
    
    ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Transmissão de Áudio com OFDM - 5s @ 16 kHz, 16 bits, Mono', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nTeste concluído!")


if __name__ == '__main__':
    main()
