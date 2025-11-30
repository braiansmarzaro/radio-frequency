"""
Transmissão de imagem RGB 24-bit (100x100 pixels) usando 16-QAM.
Objetivo: transmitir em até 3 segundos.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from entities import QAM16Transmitter, QAM16Receiver, AWGNChannel, Transmission


def generate_test_image(width: int = 100, height: int = 100) -> np.ndarray:
    """
    Gera uma imagem de teste RGB colorida.
    
    Args:
        width: Largura da imagem em pixels
        height: Altura da imagem em pixels
        
    Returns:
        Array numpy (height, width, 3) com valores 0-255
    """
    # Cria imagem com gradientes e padrões
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Gradiente vermelho
            image[y, x, 0] = int((x / width) * 255)
            
            # Gradiente verde
            image[y, x, 1] = int((y / height) * 255)
            
            # Padrão azul (xadrez)
            if (x // 10 + y // 10) % 2 == 0:
                image[y, x, 2] = 255
            else:
                image[y, x, 2] = 0
    
    # Adiciona algumas formas geométricas
    # Círculo vermelho no centro
    center_y, center_x = height // 2, width // 2
    radius = min(width, height) // 4
    
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < radius:
                image[y, x] = [255, 0, 0]  # Vermelho
    
    # Retângulo verde
    rect_y1, rect_y2 = height // 4, height // 2
    rect_x1, rect_x2 = width // 4, width // 2
    image[rect_y1:rect_y2, rect_x1:rect_x2] = [0, 255, 0]  # Verde
    
    # Retângulo azul
    rect_y1, rect_y2 = height // 2, 3 * height // 4
    rect_x1, rect_x2 = width // 2, 3 * width // 4
    image[rect_y1:rect_y2, rect_x1:rect_x2] = [0, 0, 255]  # Azul
    
    return image


def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Converte imagem numpy para bytes.
    
    Args:
        image: Array numpy da imagem
        
    Returns:
        Bytes da imagem
    """
    return image.tobytes()


def bytes_to_image(data: bytes, width: int, height: int) -> np.ndarray:
    """
    Converte bytes de volta para imagem.
    
    Args:
        data: Bytes da imagem
        width: Largura da imagem
        height: Altura da imagem
        
    Returns:
        Array numpy da imagem reconstruída
    """
    expected_size = width * height * 3
    
    # Ajusta tamanho se necessário
    if len(data) < expected_size:
        # Padding com zeros
        data = data + bytes(expected_size - len(data))
    elif len(data) > expected_size:
        # Trunca
        data = data[:expected_size]
    
    image = np.frombuffer(data, dtype=np.uint8)
    image = image.reshape((height, width, 3))
    
    return image


def calculate_required_parameters(image_size: tuple, max_duration: float = 3.0):
    """
    Calcula parâmetros necessários para transmitir imagem em tempo máximo.
    
    Args:
        image_size: (height, width, channels)
        max_duration: Duração máxima em segundos
        
    Returns:
        Dicionário com parâmetros calculados
    """
    height, width, channels = image_size
    
    # Total de bytes
    total_bytes = height * width * channels
    total_bits = total_bytes * 8
    
    # 16-QAM: 4 bits por símbolo
    bits_per_symbol = 4
    total_symbols = total_bits / bits_per_symbol
    
    # Taxa de símbolos necessária
    required_symbol_rate = total_symbols / max_duration
    
    # Taxa de amostragem: 10 amostras por símbolo para boa qualidade
    samples_per_symbol = 10
    sample_rate = required_symbol_rate * samples_per_symbol
    
    # Largura de banda aproximada (Nyquist)
    bandwidth = required_symbol_rate
    
    return {
        'total_bytes': total_bytes,
        'total_bits': total_bits,
        'total_symbols': total_symbols,
        'symbol_rate': required_symbol_rate,
        'sample_rate': sample_rate,
        'bit_rate': required_symbol_rate * bits_per_symbol,
        'samples_per_symbol': samples_per_symbol,
        'bandwidth': bandwidth
    }


def calculate_psnr(original: np.ndarray, received: np.ndarray) -> float:
    """
    Calcula PSNR (Peak Signal-to-Noise Ratio) entre imagens.
    
    Args:
        original: Imagem original
        received: Imagem recebida
        
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
    Simulação de transmissão de imagem RGB usando 16-QAM.
    """
    print("=" * 70)
    print("TRANSMISSÃO DE IMAGEM RGB - 16-QAM")
    print("=" * 70)
    
    # Parâmetros
    width, height = 100, 100
    max_duration = 3.0  # 3 segundos
    carrier_freq = 0    # Banda base (pode ser alterado)
    snr_db = 15        # 20 dB SNR
    
    print("\n--- Especificações da Imagem ---")
    print(f"Dimensões: {width}x{height} pixels")
    print(f"Formato: RGB 24-bit (3 canais)")
    print(f"Tamanho: {width * height * 3:,} bytes")
    print(f"Tempo Máximo: {max_duration} segundos")
    
    # Calcula parâmetros
    params = calculate_required_parameters((height, width, 3), max_duration)
    
    print("\n--- Parâmetros Calculados ---")
    print(f"Total de Bits: {params['total_bits']:,}")
    print(f"Total de Símbolos: {params['total_symbols']:,.0f}")
    print(f"Taxa de Símbolos: {params['symbol_rate']/1e3:.2f} kbaud")
    print(f"Taxa de Bits: {params['bit_rate']/1e3:.2f} kbps")
    print(f"Taxa de Amostragem: {params['sample_rate']/1e6:.3f} MS/s")
    print(f"Largura de Banda: {params['bandwidth']/1e3:.2f} kHz")
    print(f"Amostras/Símbolo: {params['samples_per_symbol']}")
    
    # 1. Gerar imagem de teste
    print("\n--- Gerando Imagem de Teste ---")
    image_original = generate_test_image(width, height)
    print(f"✓ Imagem gerada: {image_original.shape}")
    
    # 2. Criar transmissor 16-QAM
    print("\n--- Criando Transmissor ---")
    transmitter = QAM16Transmitter(
        sample_rate=params['sample_rate'],
        symbol_rate=params['symbol_rate'],
        carrier_freq=carrier_freq
    )
    print(f"✓ Transmissor 16-QAM criado")
    
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
    print(f"✓ Receptor 16-QAM criado")
    
    # 5. Transmitir imagem
    print("\n" + "=" * 70)
    print("INICIANDO TRANSMISSÃO")
    print("=" * 70)
    
    start_time = time.time()
    
    # Converte imagem para bytes e bits
    image_bytes = image_to_bytes(image_original)
    bits = transmitter.bits_from_bytes(image_bytes)
    
    print(f"\n1. Modulando {len(bits):,} bits...")
    tx_start = time.time()
    tx_signal = transmitter.modulate(bits)
    tx_time = time.time() - tx_start
    print(f"   ✓ Modulação concluída em {tx_time:.3f}s")
    print(f"   Sinal transmitido: {len(tx_signal):,} amostras complexas")
    
    # Duração do sinal
    signal_duration = len(tx_signal) / params['sample_rate']
    print(f"   Duração do sinal: {signal_duration:.3f}s")
    
    print(f"\n2. Transmitindo pelo canal AWGN...")
    ch_start = time.time()
    rx_signal = channel.transmit(tx_signal)
    ch_time = time.time() - ch_start
    print(f"   ✓ Canal aplicado em {ch_time:.3f}s")
    
    print(f"\n3. Demodulando sinal recebido...")
    rx_start = time.time()
    rx_bits = receiver.demodulate(rx_signal, use_rrc=False)
    rx_time = time.time() - rx_start
    print(f"   ✓ Demodulação concluída em {rx_time:.3f}s")
    
    # Ajusta comprimento
    min_len = min(len(bits), len(rx_bits))
    rx_bits = rx_bits[:min_len]
    original_bits = bits[:min_len]
    
    print(f"\n4. Reconstruindo imagem...")
    rx_bytes = receiver.bytes_from_bits(rx_bits)
    image_received = bytes_to_image(rx_bytes, width, height)
    print(f"   ✓ Imagem reconstruída: {image_received.shape}")
    
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
    psnr = calculate_psnr(image_original, image_received)
    print(f"\nPSNR: {psnr:.2f} dB")
    
    # MSE por canal
    mse_r = np.mean((image_original[:,:,0].astype(float) - image_received[:,:,0].astype(float))**2)
    mse_g = np.mean((image_original[:,:,1].astype(float) - image_received[:,:,1].astype(float))**2)
    mse_b = np.mean((image_original[:,:,2].astype(float) - image_received[:,:,2].astype(float))**2)
    
    print(f"\nMSE por Canal:")
    print(f"  Red:   {mse_r:.2f}")
    print(f"  Green: {mse_g:.2f}")
    print(f"  Blue:  {mse_b:.2f}")
    
    # SNR
    signal_power = np.mean(np.abs(tx_signal)**2)
    noise = rx_signal - tx_signal
    noise_power = np.mean(np.abs(noise)**2)
    snr_measured = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    print(f"\nSNR configurada: {snr_db} dB")
    print(f"SNR medida: {snr_measured:.2f} dB")
    
    # Throughput
    throughput = (len(original_bits) * (1 - ber)) / signal_duration
    print(f"\nThroughput efetivo: {throughput/1e3:.2f} kbps")
    
    # 8. Visualização
    print("\n" + "=" * 70)
    print("GERANDO VISUALIZAÇÕES")
    print("=" * 70)
    
    # Figura com 3 subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Imagem Original
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image_original)
    ax1.set_title('Imagem Original', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    # Subplot 2: Imagem Recebida
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(image_received)
    ax2.set_title(f'Imagem Recebida (PSNR: {psnr:.2f} dB)', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    # Subplot 3: Diferença (erro)
    ax3 = plt.subplot(2, 3, 3)
    diff = np.abs(image_original.astype(float) - image_received.astype(float))
    ax3.imshow(diff.astype(np.uint8))
    ax3.set_title('Erro Absoluto (Diferença)', fontsize=13, fontweight='bold')
    ax3.axis('off')
    
    # Subplot 4: Diagrama de Constelação
    ax4 = plt.subplot(2, 3, 4)
    
    # Amostra símbolos recebidos
    baseband_rx = receiver.downconvert(rx_signal)
    baseband_rx = receiver.apply_agc(baseband_rx)
    symbols = baseband_rx[::receiver.samples_per_symbol][:1000]
    
    ax4.scatter(np.real(symbols), np.imag(symbols), s=10, alpha=0.5, c='blue', label='Recebidos')
    ax4.scatter(np.real(transmitter.constellation), np.imag(transmitter.constellation),
               s=200, c='red', marker='x', linewidths=3, label='Ideal')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.axvline(x=0, color='k', linewidth=0.5)
    ax4.set_xlabel('In-Phase (I)', fontsize=11)
    ax4.set_ylabel('Quadrature (Q)', fontsize=11)
    ax4.set_title(f'Diagrama de Constelação 16-QAM', fontsize=13, fontweight='bold')
    ax4.axis('equal')
    ax4.legend(fontsize=9)
    
    # Subplot 5: Histograma de erros de pixel
    ax5 = plt.subplot(2, 3, 5)
    pixel_errors = np.abs(image_original.astype(int) - image_received.astype(int))
    ax5.hist(pixel_errors.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax5.set_xlabel('Erro de Pixel (0-255)', fontsize=11)
    ax5.set_ylabel('Frequência', fontsize=11)
    ax5.set_title('Distribuição de Erros de Pixel', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Comparação RGB lado a lado
    ax6 = plt.subplot(2, 3, 6)
    
    # Cria faixas com os canais
    strip_height = height // 3
    comparison = np.zeros_like(image_original)
    
    # Original R, Recebido R
    comparison[:strip_height, :width//2, 0] = image_original[:strip_height, :width//2, 0]
    comparison[:strip_height, width//2:, 0] = image_received[:strip_height, width//2:, 0]
    
    # Original G, Recebido G
    comparison[strip_height:2*strip_height, :width//2, 1] = image_original[strip_height:2*strip_height, :width//2, 1]
    comparison[strip_height:2*strip_height, width//2:, 1] = image_received[strip_height:2*strip_height, width//2:, 1]
    
    # Original B, Recebido B
    comparison[2*strip_height:, :width//2, 2] = image_original[2*strip_height:, :width//2, 2]
    comparison[2*strip_height:, width//2:, 2] = image_received[2*strip_height:, width//2:, 2]
    
    ax6.imshow(comparison)
    ax6.set_title('Comparação por Canal (Orig | Rec)', fontsize=13, fontweight='bold')
    ax6.axis('off')
    
    # Adiciona linhas divisórias
    ax6.axhline(y=strip_height, color='white', linewidth=2)
    ax6.axhline(y=2*strip_height, color='white', linewidth=2)
    ax6.axvline(x=width//2, color='white', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualizações exibidas")
    
    # 9. Resumo
    print("\n" + "=" * 70)
    print("TRANSMISSÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    
    print(f"\n📊 RESUMO:")
    print(f"   • Imagem: {width}x{height} RGB ({params['total_bytes']:,} bytes)")
    print(f"   • Duração: {signal_duration:.3f}s (limite: {max_duration}s)")
    print(f"   • BER: {ber:.2e}")
    print(f"   • PSNR: {psnr:.2f} dB")
    print(f"   • Modulação: 16-QAM")
    print(f"   • Largura de Banda: {params['bandwidth']/1e3:.2f} kHz")


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
