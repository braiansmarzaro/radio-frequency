"""
Transmissão de texto longo (10k caracteres) usando 4-ASK com portadora em 450 MHz.
Objetivo: transmitir em até 2 segundos.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from entities import ASK4Transmitter, ASK4Receiver, AWGNChannel, Transmission


def generate_long_text(length: int = 10000) -> str:
    """
    Gera texto de exemplo com comprimento específico.
    
    Args:
        length: Comprimento do texto em caracteres
        
    Returns:
        String com o comprimento desejado
    """
    base_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia. "
    )
    
    # Repete o texto base até atingir o comprimento desejado
    repetitions = (length // len(base_text)) + 1
    long_text = (base_text * repetitions)[:length]
    
    return long_text


def calculate_required_parameters(text_length: int, max_duration: float = 2.0):
    """
    Calcula parâmetros necessários para transmitir em tempo máximo.
    
    Args:
        text_length: Comprimento do texto em caracteres
        max_duration: Duração máxima em segundos
        
    Returns:
        Dicionário com parâmetros calculados
    """
    # Cada caractere UTF-8 = 8 bits
    total_bits = text_length * 8
    
    # 4-ASK: 2 bits por símbolo
    bits_per_symbol = 2
    total_symbols = total_bits / bits_per_symbol
    
    # Taxa de símbolos necessária para transmitir em max_duration
    required_symbol_rate = total_symbols / max_duration
    
    # Taxa de amostragem: pelo menos 2x a taxa de símbolos (Nyquist)
    # Usamos um fator maior para melhor qualidade
    sample_rate = required_symbol_rate * 10  # 10 amostras por símbolo
    
    return {
        'total_bits': total_bits,
        'total_symbols': total_symbols,
        'symbol_rate': required_symbol_rate,
        'sample_rate': sample_rate,
        'bit_rate': required_symbol_rate * bits_per_symbol,
        'samples_per_symbol': 10
    }


def main():
    """
    Simulação de transmissão de texto longo usando 4-ASK em 450 MHz.
    """
    print("=" * 70)
    print("TRANSMISSÃO DE TEXTO LONGO - 4-ASK @ 450 MHz")
    print("=" * 70)
    
    # Parâmetros do desafio
    text_length = 10000  # 10k caracteres
    max_duration = 2.0   # 2 segundos
    carrier_freq = 450e6  # 450 MHz
    snr_db = 15          # 20 dB SNR (bom para demonstração)
    
    # Calcula parâmetros necessários
    params = calculate_required_parameters(text_length, max_duration)
    
    print("\n--- Requisitos da Transmissão ---")
    print(f"Texto: {text_length} caracteres")
    print(f"Tempo Máximo: {max_duration} segundos")
    print(f"Frequência Portadora: {carrier_freq/1e6:.0f} MHz")
    
    print("\n--- Parâmetros Calculados ---")
    print(f"Total de Bits: {params['total_bits']:,}")
    print(f"Total de Símbolos: {params['total_symbols']:,.0f}")
    print(f"Taxa de Símbolos: {params['symbol_rate']/1e3:.2f} kbaud")
    print(f"Taxa de Bits: {params['bit_rate']/1e3:.2f} kbps")
    print(f"Taxa de Amostragem: {params['sample_rate']/1e6:.2f} MS/s")
    print(f"Amostras/Símbolo: {params['samples_per_symbol']}")
    
    # 1. Criar transmissor 4-ASK
    print("\n--- Criando Transmissor ---")
    transmitter = ASK4Transmitter(
        sample_rate=params['sample_rate'],
        symbol_rate=params['symbol_rate'],
        carrier_freq=carrier_freq
    )
    print(f"✓ Transmissor 4-ASK criado")
    print(f"  Eficiência Espectral: {transmitter.get_spectral_efficiency()} bits/s/Hz")
    
    # 2. Criar canal AWGN
    print("\n--- Criando Canal ---")
    channel = AWGNChannel(snr_db=snr_db)
    print(f"✓ Canal AWGN criado (SNR = {snr_db} dB)")
    
    # 3. Criar receptor 4-ASK
    print("\n--- Criando Receptor ---")
    receiver = ASK4Receiver(
        sample_rate=params['sample_rate'],
        symbol_rate=params['symbol_rate'],
        carrier_freq=carrier_freq
    )
    print(f"✓ Receptor 4-ASK criado")
    
    # 4. Criar sistema de transmissão
    system = Transmission(transmitter, channel, receiver)
    
    # 5. Gerar texto
    print("\n--- Gerando Texto ---")
    texto_original = generate_long_text(text_length)
    print(f"✓ Texto gerado: {len(texto_original)} caracteres")
    print(f"\nPrimeiros 100 caracteres:")
    print(f'"{texto_original[:100]}..."')
    
    # 6. Transmitir
    print("\n" + "=" * 70)
    print("INICIANDO TRANSMISSÃO")
    print("=" * 70)
    
    start_time = time.time()
    
    # Converte texto para bits
    data_bytes = texto_original.encode('utf-8')
    bits = transmitter.bits_from_bytes(data_bytes)
    
    print(f"\n1. Modulando {len(bits):,} bits...")
    tx_start = time.time()
    tx_signal = transmitter.modulate(bits)
    tx_time = time.time() - tx_start
    print(f"   ✓ Modulação concluída em {tx_time:.3f}s")
    print(f"   Sinal transmitido: {len(tx_signal):,} amostras complexas")
    
    # Duração teórica do sinal
    signal_duration = len(tx_signal) / params['sample_rate']
    print(f"   Duração do sinal: {signal_duration:.3f}s")
    
    print(f"\n2. Transmitindo pelo canal AWGN...")
    ch_start = time.time()
    rx_signal = channel.transmit(tx_signal)
    ch_time = time.time() - ch_start
    print(f"   ✓ Canal aplicado em {ch_time:.3f}s")
    
    print(f"\n3. Demodulando sinal recebido...")
    rx_start = time.time()
    rx_bits = receiver.demodulate(rx_signal)
    rx_time = time.time() - rx_start
    print(f"   ✓ Demodulação concluída em {rx_time:.3f}s")
    
    # Ajusta comprimento
    min_len = min(len(bits), len(rx_bits))
    rx_bits = rx_bits[:min_len]
    original_bits = bits[:min_len]
    
    print(f"\n4. Decodificando bits em texto...")
    rx_bytes = receiver.bytes_from_bits(rx_bits)
    
    try:
        texto_recebido = rx_bytes.decode('utf-8', errors='ignore')
    except:
        texto_recebido = "[Erro na decodificação]"
    
    total_time = time.time() - start_time
    
    print(f"   ✓ Decodificação concluída")
    
    # 7. Resultados
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
    
    print(f"\n--- Texto Recebido ---")
    print(f"Comprimento: {len(texto_recebido)} caracteres")
    print(f"\nPrimeiros 100 caracteres:")
    print(f'"{texto_recebido[:100]}..."')
    
    print(f"\nÚltimos 100 caracteres:")
    print(f'"...{texto_recebido[-100:]}"')
    
    # 8. Métricas de desempenho
    print(f"\n--- Métricas de Desempenho ---")
    
    # BER
    bit_errors = np.sum(original_bits != rx_bits)
    ber = bit_errors / len(original_bits)
    print(f"BER: {ber:.2e}")
    print(f"Bits com erro: {bit_errors:,} / {len(original_bits):,}")
    
    # Comparação de caracteres
    min_char_len = min(len(texto_original), len(texto_recebido))
    char_errors = sum(1 for i in range(min_char_len) 
                     if texto_original[i] != texto_recebido[i])
    char_error_rate = char_errors / min_char_len if min_char_len > 0 else 0
    
    print(f"\nCaracteres corretos: {min_char_len - char_errors:,} / {min_char_len:,}")
    print(f"Taxa de erro de caracteres: {char_error_rate:.2%}")
    
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
    print(f"Taxa de bits nominal: {params['bit_rate']/1e3:.2f} kbps")
    
    # BER teórica (ASK não está implementada na teoria, usando aproximação)
    print(f"\n--- Informações do Sistema ---")
    print(f"Modulação: 4-ASK")
    print(f"Portadora: {carrier_freq/1e6:.0f} MHz")
    print(f"Banda aproximada: {params['symbol_rate']/1e6:.2f} MHz")
    print(f"Potência do sinal: {10*np.log10(signal_power):.2f} dBm (relativo)")
    
    # 9. Estatísticas de erros
    if char_errors > 0 and char_errors <= 20:
        print(f"\n--- Primeiros {min(char_errors, 20)} Erros de Caracteres ---")
        error_count = 0
        for i in range(min_char_len):
            if texto_original[i] != texto_recebido[i]:
                print(f"Posição {i}: '{texto_original[i]}' → '{texto_recebido[i]}'")
                error_count += 1
                if error_count >= 20:
                    break
    
    print("\n" + "=" * 70)
    print("TRANSMISSÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    
    # 10. Resumo final
    print(f"\n📊 RESUMO:")
    print(f"   • Texto: {len(texto_original):,} caracteres transmitidos")
    print(f"   • Duração: {signal_duration:.3f}s (limite: {max_duration}s)")
    print(f"   • Taxa: {len(texto_original)/signal_duration:.0f} caracteres/segundo")
    print(f"   • BER: {ber:.2e}")
    print(f"   • Acurácia: {(1-char_error_rate)*100:.2f}%")
    print(f"   • Portadora: {carrier_freq/1e6:.0f} MHz")
    
    # 11. Gráfico de constelação
    print("\n" + "=" * 70)
    print("GERANDO GRÁFICO DE CONSTELAÇÃO")
    print("=" * 70)
    
    # Extrai símbolos transmitidos e recebidos
    # Para símbolos transmitidos, precisamos amostrar diretamente do sinal modulado
    baseband_rx = receiver.downconvert(rx_signal)
    baseband_rx = receiver.apply_agc(baseband_rx)
    
    # Amostra símbolos recebidos
    num_symbols_to_plot = min(1000, len(baseband_rx) // receiver.samples_per_symbol)
    rx_symbols = np.real(baseband_rx[::receiver.samples_per_symbol][:num_symbols_to_plot])
    
    # Para símbolos transmitidos, fazemos downconvert do sinal TX
    baseband_tx = rx_signal  # Usamos o sinal antes do canal para referência
    if carrier_freq > 0:
        t = np.arange(len(tx_signal)) / transmitter.sample_rate
        carrier = np.exp(-1j * 2 * np.pi * carrier_freq * t)
        baseband_tx = tx_signal * carrier
    else:
        baseband_tx = tx_signal
    
    tx_symbols = np.real(baseband_tx[::transmitter.samples_per_symbol][:num_symbols_to_plot])
    
    # Cria figura com 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Histograma dos símbolos
    axes[0].hist(rx_symbols, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Símbolos Recebidos')
    
    # Marca níveis ideais
    for i, level in enumerate(transmitter.amplitude_levels):
        axes[0].axvline(x=level, color='red', linestyle='--', linewidth=2, 
                       label='Níveis Ideais' if i == 0 else '')
    
    # Marca limiares de decisão
    for i, threshold in enumerate(receiver.decision_thresholds):
        axes[0].axvline(x=threshold, color='green', linestyle=':', linewidth=1.5,
                       label='Limiares' if i == 0 else '')
    
    axes[0].set_xlabel('Amplitude', fontsize=12)
    axes[0].set_ylabel('Frequência', fontsize=12)
    axes[0].set_title('Histograma de Símbolos Recebidos - 4-ASK', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Subplot 2: Dispersão temporal dos símbolos
    sample_indices = range(len(tx_symbols))
    
    axes[1].scatter(sample_indices, rx_symbols, s=8, alpha=0.5, c='blue', label='Recebidos')
    axes[1].scatter(sample_indices, tx_symbols, s=8, alpha=0.3, c='orange', label='Transmitidos')
    
    # Níveis ideais
    for level in transmitter.amplitude_levels:
        axes[1].axhline(y=level, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    axes[1].set_xlabel('Índice do Símbolo', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_title(f'Dispersão de Símbolos ({num_symbols_to_plot} amostras)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim([0, min(500, num_symbols_to_plot)])  # Limita visualização para clareza
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Gráfico de constelação exibido")


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
