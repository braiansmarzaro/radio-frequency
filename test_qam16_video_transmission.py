"""
Transmissão de vídeo em escala de cinza (10 frames de 100x100) usando 16-QAM.
Objetivo: transmitir em até 1 segundo.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from entities import QAM16Transmitter, QAM16Receiver, AWGNChannel, Transmission


def generate_test_video(num_frames: int = 10, width: int = 100, height: int = 100) -> np.ndarray:
    """
    Gera vídeo de teste em escala de cinza com movimento.
    
    Args:
        num_frames: Número de frames
        width: Largura em pixels
        height: Altura em pixels
        
    Returns:
        Array numpy (num_frames, height, width) com valores 0-255
    """
    video = np.zeros((num_frames, height, width), dtype=np.uint8)
    
    for frame_idx in range(num_frames):
        # Cria frame com padrões que mudam ao longo do tempo
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Gradiente que muda com o tempo
        for y in range(height):
            for x in range(width):
                # Gradiente diagonal rotativo
                angle = (frame_idx / num_frames) * 2 * np.pi
                gradient = (x * np.cos(angle) + y * np.sin(angle)) / (width + height)
                frame[y, x] = int((gradient + 0.5) * 255) % 256
        
        # Adiciona círculo em movimento
        center_x = int(width // 2 + (width // 4) * np.cos(2 * np.pi * frame_idx / num_frames))
        center_y = int(height // 2 + (height // 4) * np.sin(2 * np.pi * frame_idx / num_frames))
        radius = width // 8
        
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < radius:
                    frame[y, x] = 255  # Círculo branco
        
        # Adiciona retângulo em movimento
        rect_x = int((frame_idx / num_frames) * width)
        rect_y = height // 2
        rect_w, rect_h = width // 10, height // 10
        
        x1, x2 = max(0, rect_x), min(width, rect_x + rect_w)
        y1, y2 = max(0, rect_y), min(height, rect_y + rect_h)
        frame[y1:y2, x1:x2] = 128  # Retângulo cinza
        
        video[frame_idx] = frame
    
    return video


def video_to_bytes(video: np.ndarray) -> bytes:
    """
    Converte vídeo numpy para bytes.
    
    Args:
        video: Array numpy do vídeo
        
    Returns:
        Bytes do vídeo
    """
    return video.tobytes()


def bytes_to_video(data: bytes, num_frames: int, width: int, height: int) -> np.ndarray:
    """
    Converte bytes de volta para vídeo.
    
    Args:
        data: Bytes do vídeo
        num_frames: Número de frames
        width: Largura
        height: Altura
        
    Returns:
        Array numpy do vídeo reconstruído
    """
    expected_size = num_frames * width * height
    
    # Ajusta tamanho se necessário
    if len(data) < expected_size:
        data = data + bytes(expected_size - len(data))
    elif len(data) > expected_size:
        data = data[:expected_size]
    
    video = np.frombuffer(data, dtype=np.uint8)
    video = video.reshape((num_frames, height, width))
    
    return video


def calculate_required_parameters(video_size: tuple, max_duration: float = 1.0):
    """
    Calcula parâmetros necessários para transmitir vídeo em tempo máximo.
    
    Args:
        video_size: (num_frames, height, width)
        max_duration: Duração máxima em segundos
        
    Returns:
        Dicionário com parâmetros calculados
    """
    num_frames, height, width = video_size
    
    # Total de bytes (escala de cinza = 1 byte por pixel)
    total_bytes = num_frames * height * width
    total_bits = total_bytes * 8
    
    # 16-QAM: 4 bits por símbolo
    bits_per_symbol = 4
    total_symbols = total_bits / bits_per_symbol
    
    # Taxa de símbolos necessária
    required_symbol_rate = total_symbols / max_duration
    
    # Taxa de amostragem
    samples_per_symbol = 10
    sample_rate = required_symbol_rate * samples_per_symbol
    
    # Largura de banda
    bandwidth = required_symbol_rate
    
    return {
        'total_bytes': total_bytes,
        'total_bits': total_bits,
        'total_symbols': total_symbols,
        'symbol_rate': required_symbol_rate,
        'sample_rate': sample_rate,
        'bit_rate': required_symbol_rate * bits_per_symbol,
        'samples_per_symbol': samples_per_symbol,
        'bandwidth': bandwidth,
        'frame_rate': num_frames / max_duration
    }


def calculate_psnr(original: np.ndarray, received: np.ndarray) -> float:
    """
    Calcula PSNR entre vídeos.
    
    Args:
        original: Vídeo original
        received: Vídeo recebido
        
    Returns:
        PSNR em dB
    """
    mse = np.mean((original.astype(float) - received.astype(float))**2)
    
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def main():
    """
    Simulação de transmissão de vídeo em escala de cinza usando 16-QAM.
    """
    print("=" * 70)
    print("TRANSMISSÃO DE VÍDEO EM ESCALA DE CINZA - 16-QAM")
    print("=" * 70)
    
    # Parâmetros
    num_frames = 10
    width, height = 100, 100
    max_duration = 1.0  # 1 segundo
    carrier_freq = 0    # Banda base
    snr_db = 25        # 25 dB SNR (alto para vídeo)
    
    print("\n--- Especificações do Vídeo ---")
    print(f"Frames: {num_frames}")
    print(f"Resolução: {width}x{height} pixels")
    print(f"Formato: Escala de cinza (8-bit)")
    print(f"Tamanho: {num_frames * width * height:,} bytes")
    print(f"Tempo Máximo: {max_duration} segundo(s)")
    
    # Calcula parâmetros
    params = calculate_required_parameters((num_frames, height, width), max_duration)
    
    print("\n--- Parâmetros Calculados ---")
    print(f"Total de Bits: {params['total_bits']:,}")
    print(f"Total de Símbolos: {params['total_symbols']:,.0f}")
    print(f"Taxa de Símbolos: {params['symbol_rate']/1e6:.3f} Mbaud")
    print(f"Taxa de Bits: {params['bit_rate']/1e6:.3f} Mbps")
    print(f"Taxa de Amostragem: {params['sample_rate']/1e6:.2f} MS/s")
    print(f"Largura de Banda: {params['bandwidth']/1e6:.3f} MHz")
    print(f"Frame Rate: {params['frame_rate']:.1f} fps")
    
    # 1. Gerar vídeo de teste
    print("\n--- Gerando Vídeo de Teste ---")
    video_original = generate_test_video(num_frames, width, height)
    print(f"✓ Vídeo gerado: {video_original.shape}")
    
    # 2. Criar transmissor 16-QAM
    print("\n--- Criando Transmissor ---")
    transmitter = QAM16Transmitter(
        sample_rate=params['sample_rate'],
        symbol_rate=params['symbol_rate'],
        carrier_freq=carrier_freq
    )
    print("✓ Transmissor 16-QAM criado")
    
    # 3. Criar canal AWGN
    print("\n--- Criando Canal ---")
    channel = AWGNChannel(snr_db=snr_db)
    print(f"✓ Canal AWGN criado (SNR = {snr_db} dB)")
    
    # 4. Criar receptor 16-QAM
    print("\n--- Criando Receptor ---")
    receiver = QAM16Receiver(
        sample_rate=params['sample_rate'],
        symbol_rate=params['symbol_rate'],
        carrier_freq=carrier_freq
    )
    print("✓ Receptor 16-QAM criado")
    
    # 5. Transmitir vídeo
    print("\n" + "=" * 70)
    print("INICIANDO TRANSMISSÃO")
    print("=" * 70)
    
    start_time = time.time()
    
    # Converte vídeo para bytes e bits
    video_bytes = video_to_bytes(video_original)
    bits = transmitter.bits_from_bytes(video_bytes)
    
    print(f"\n1. Modulando {len(bits):,} bits...")
    tx_start = time.time()
    tx_signal = transmitter.modulate(bits)
    tx_time = time.time() - tx_start
    print(f"   ✓ Modulação concluída em {tx_time:.3f}s")
    print(f"   Sinal transmitido: {len(tx_signal):,} amostras complexas")
    
    # Duração do sinal
    signal_duration = len(tx_signal) / params['sample_rate']
    print(f"   Duração do sinal: {signal_duration:.3f}s")
    
    print("\n2. Transmitindo pelo canal AWGN...")
    ch_start = time.time()
    rx_signal = channel.transmit(tx_signal)
    ch_time = time.time() - ch_start
    print(f"   ✓ Canal aplicado em {ch_time:.3f}s")
    
    print("\n3. Demodulando sinal recebido...")
    rx_start = time.time()
    rx_bits = receiver.demodulate(rx_signal, use_rrc=False)
    rx_time = time.time() - rx_start
    print(f"   ✓ Demodulação concluída em {rx_time:.3f}s")
    
    # Ajusta comprimento
    min_len = min(len(bits), len(rx_bits))
    rx_bits = rx_bits[:min_len]
    original_bits = bits[:min_len]
    
    print("\n4. Reconstruindo vídeo...")
    rx_bytes = receiver.bytes_from_bits(rx_bits)
    video_received = bytes_to_video(rx_bytes, num_frames, width, height)
    print(f"   ✓ Vídeo reconstruído: {video_received.shape}")
    
    total_time = time.time() - start_time
    
    # 6. Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    
    print(f"\n--- Tempos ---")
    print(f"Processamento Total: {total_time:.3f}s")
    print(f"  - Modulação:       {tx_time:.3f}s")
    print(f"  - Canal:           {ch_time:.3f}s")
    print(f"  - Demodulação:     {rx_time:.3f}s")
    print(f"Duração do Sinal:   {signal_duration:.3f}s")
    print(f"Objetivo:           {max_duration:.3f}s")
    
    if signal_duration <= max_duration:
        print(f"✓ OBJETIVO ATINGIDO! ({signal_duration:.3f}s ≤ {max_duration:.3f}s)")
    else:
        print(f"✗ Objetivo não atingido ({signal_duration:.3f}s > {max_duration:.3f}s)")
    
    # 7. Métricas de qualidade
    print(f"\n--- Métricas de Qualidade ---")
    
    # BER
    bit_errors = np.sum(original_bits != rx_bits)
    ber = bit_errors / len(original_bits)
    print(f"BER: {ber:.2e}")
    print(f"Bits com erro: {bit_errors:,} / {len(original_bits):,}")
    
    # PSNR
    psnr = calculate_psnr(video_original, video_received)
    print(f"\nPSNR Global: {psnr:.2f} dB")
    
    # PSNR por frame
    print("\nPSNR por Frame:")
    psnr_per_frame = []
    for i in range(num_frames):
        psnr_frame = calculate_psnr(video_original[i], video_received[i])
        psnr_per_frame.append(psnr_frame)
        print(f"  Frame {i+1:2d}: {psnr_frame:.2f} dB")
    
    # SNR
    signal_power = np.mean(np.abs(tx_signal)**2)
    noise = rx_signal - tx_signal
    noise_power = np.mean(np.abs(noise)**2)
    snr_measured = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    print(f"\nSNR configurada: {snr_db} dB")
    print(f"SNR medida: {snr_measured:.2f} dB")
    
    # Throughput
    throughput = (len(original_bits) * (1 - ber)) / signal_duration
    print(f"\nThroughput efetivo: {throughput/1e6:.3f} Mbps")
    print(f"Taxa de bits nominal: {params['bit_rate']/1e6:.3f} Mbps")
    
    # 8. Visualização
    print("\n" + "=" * 70)
    print("GERANDO VISUALIZAÇÕES")
    print("=" * 70)
    
    # Figura principal
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 3 linhas x 4 colunas
    # Linha 1: Frames originais selecionados
    # Linha 2: Frames recebidos correspondentes
    # Linha 3: Gráficos de métricas
    
    frames_to_show = [0, 3, 6, 9]  # 4 frames distribuídos
    
    for idx, frame_idx in enumerate(frames_to_show):
        # Frame original
        ax_orig = plt.subplot(3, 4, idx + 1)
        ax_orig.imshow(video_original[frame_idx], cmap='gray', vmin=0, vmax=255)
        ax_orig.set_title(f'Original - Frame {frame_idx+1}', fontsize=11, fontweight='bold')
        ax_orig.axis('off')
        
        # Frame recebido
        ax_recv = plt.subplot(3, 4, idx + 5)
        ax_recv.imshow(video_received[frame_idx], cmap='gray', vmin=0, vmax=255)
        ax_recv.set_title(f'Recebido - Frame {frame_idx+1}\nPSNR: {psnr_per_frame[frame_idx]:.1f} dB', 
                         fontsize=11, fontweight='bold')
        ax_recv.axis('off')
    
    # Subplot: PSNR por frame
    ax_psnr = plt.subplot(3, 4, 9)
    ax_psnr.plot(range(1, num_frames+1), psnr_per_frame, 'bo-', linewidth=2, markersize=8)
    ax_psnr.axhline(y=np.mean(psnr_per_frame), color='r', linestyle='--', 
                    label=f'Média: {np.mean(psnr_per_frame):.2f} dB')
    ax_psnr.set_xlabel('Frame', fontsize=11)
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=11)
    ax_psnr.set_title('Qualidade por Frame', fontsize=11, fontweight='bold')
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.legend(fontsize=9)
    ax_psnr.set_xticks(range(1, num_frames+1))
    
    # Subplot: Diagrama de constelação
    ax_const = plt.subplot(3, 4, 10)
    baseband_rx = receiver.downconvert(rx_signal)
    baseband_rx = receiver.apply_agc(baseband_rx)
    symbols = baseband_rx[::receiver.samples_per_symbol][:1000]
    
    ax_const.scatter(np.real(symbols), np.imag(symbols), s=8, alpha=0.4, c='blue', label='Recebidos')
    ax_const.scatter(np.real(transmitter.constellation), np.imag(transmitter.constellation),
                    s=150, c='red', marker='x', linewidths=2, label='Ideal')
    ax_const.grid(True, alpha=0.3)
    ax_const.axhline(y=0, color='k', linewidth=0.5)
    ax_const.axvline(x=0, color='k', linewidth=0.5)
    ax_const.set_xlabel('In-Phase (I)', fontsize=10)
    ax_const.set_ylabel('Quadrature (Q)', fontsize=10)
    ax_const.set_title('Constelação 16-QAM', fontsize=11, fontweight='bold')
    ax_const.axis('equal')
    ax_const.legend(fontsize=9)
    
    # Subplot: Histograma de erros de pixel
    ax_hist = plt.subplot(3, 4, 11)
    pixel_errors = np.abs(video_original.astype(int) - video_received.astype(int))
    ax_hist.hist(pixel_errors.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax_hist.set_xlabel('Erro de Pixel (0-255)', fontsize=10)
    ax_hist.set_ylabel('Frequência', fontsize=10)
    ax_hist.set_title('Distribuição de Erros', fontsize=11, fontweight='bold')
    ax_hist.grid(True, alpha=0.3)
    
    # Subplot: Informações
    ax_info = plt.subplot(3, 4, 12)
    ax_info.axis('off')
    info_text = (
        f"INFORMAÇÕES DO VÍDEO\n"
        f"{'='*30}\n\n"
        f"Frames: {num_frames}\n"
        f"Resolução: {width}×{height}\n"
        f"Tamanho: {params['total_bytes']/1024:.1f} KB\n\n"
        f"TRANSMISSÃO\n"
        f"{'='*30}\n\n"
        f"Modulação: 16-QAM\n"
        f"Taxa: {params['bit_rate']/1e6:.2f} Mbps\n"
        f"Duração: {signal_duration:.3f}s\n"
        f"BER: {ber:.2e}\n"
        f"PSNR: {psnr:.2f} dB\n"
        f"SNR: {snr_db} dB\n\n"
        f"Status: {'✓ OK' if signal_duration <= max_duration else '✗ FALHA'}"
    )
    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualizações exibidas")
    
    # 9. Resumo
    print("\n" + "=" * 70)
    print("TRANSMISSÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    
    print(f"\n📊 RESUMO:")
    print(f"   • Vídeo: {num_frames} frames {width}x{height} ({params['total_bytes']:,} bytes)")
    print(f"   • Duração: {signal_duration:.3f}s (limite: {max_duration}s)")
    print(f"   • Frame Rate: {num_frames/signal_duration:.1f} fps")
    print(f"   • BER: {ber:.2e}")
    print(f"   • PSNR Médio: {np.mean(psnr_per_frame):.2f} dB")
    print(f"   • Modulação: 16-QAM")
    print(f"   • Largura de Banda: {params['bandwidth']/1e6:.3f} MHz")


if __name__ == "__main__":
    # Define seed para reprodutibilidade
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTransmissão interrompida pelo usuário.")
    except Exception as e:
        print(f"\n\nErro durante a transmissão: {e}")
        import traceback
        traceback.print_exc()
