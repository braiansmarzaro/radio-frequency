"""
Exemplo de transmissão de texto usando modulação 16-QAM.
Demonstra o uso completo do sistema: transmissor, canal e receptor.
"""
import numpy as np
import matplotlib.pyplot as plt
from entities import QAM16Transmitter, QAM16Receiver, AWGNChannel, Transmission


def main():
    """
    Simula transmissão de texto usando 16-QAM através de canal AWGN.
    """
    print("=" * 60)
    print("SIMULAÇÃO DE TRANSMISSÃO DE TEXTO - 16-QAM")
    print("=" * 60)
    
    # Parâmetros do sistema
    sample_rate = 1e6      # 1 MS/s
    symbol_rate = 10e3     # 10 kbaud
    carrier_freq = 450e6       # Banda base
    snr_db = 15           # 15 dB SNR
    
    print("\n--- Configuração do Sistema ---")
    print(f"Taxa de Amostragem: {sample_rate/1e6:.2f} MS/s")
    print(f"Taxa de Símbolos: {symbol_rate/1e3:.2f} kbaud")
    print(f"Taxa de Bits: {symbol_rate * 4 / 1e3:.2f} kbps (4 bits/símbolo)")
    print(f"SNR do Canal: {snr_db} dB")
    
    # 1. Criar transmissor 16-QAM
    transmitter = QAM16Transmitter(
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        carrier_freq=carrier_freq
    )
    
    print(f"\n{transmitter}")
    
    # 2. Criar canal AWGN
    channel = AWGNChannel(snr_db=snr_db)
    
    print(f"\n{channel}")
    
    # 3. Criar receptor 16-QAM
    receiver = QAM16Receiver(
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        carrier_freq=carrier_freq
    )
    
    print(f"\n{receiver}")
    
    # 4. Criar sistema de transmissão
    system = Transmission(transmitter, channel, receiver)
    
    # 5. Texto a ser transmitido
    texto_original = "Hello World! Teste de transmissao 16-QAM em canal AWGN."
    
    print("\n" + "=" * 60)
    print("TRANSMISSÃO")
    print("=" * 60)
    print(f"\nTexto Original ({len(texto_original)} caracteres):")
    print(f'"{texto_original}"')
    
    # 6. Transmitir texto
    texto_recebido, metricas = system.transmit_text(texto_original)
    
    print(f"\nTexto Recebido ({len(texto_recebido)} caracteres):")
    print(f'"{texto_recebido}"')
    
    # 7. Exibir métricas
    print("\n" + "=" * 60)
    print("MÉTRICAS DE DESEMPENHO")
    print("=" * 60)
    print(f"BER (Bit Error Rate): {metricas['ber']:.2e}")
    print(f"Total de Bits: {metricas['total_bits']}")
    print(f"Bits com Erro: {metricas['error_bits']}")
    
    # Medir SNR
    snr_medida = system.measure_snr()
    print(f"SNR Medida: {snr_medida:.2f} dB")
    
    # Calcular throughput
    throughput = system.calculate_throughput()
    print(f"Throughput Efetivo: {throughput/1e3:.2f} kbps")
    
    # BER teórica
    ber_teorica = channel.get_theoretical_ber('16QAM')
    print(f"BER Teórica (16-QAM, SNR={snr_db}dB): {ber_teorica:.2e}")
    
    # 8. Comparação de caracteres
    print("\n" + "=" * 60)
    print("ANÁLISE DE CARACTERES")
    print("=" * 60)
    
    min_len = min(len(texto_original), len(texto_recebido))
    erros_char = 0
    
    for i in range(min_len):
        if texto_original[i] != texto_recebido[i]:
            erros_char += 1
            print(f"Posição {i}: '{texto_original[i]}' → '{texto_recebido[i]}'")
    
    taxa_erro_char = erros_char / min_len if min_len > 0 else 0
    print(f"\nCaracteres Corretos: {min_len - erros_char}/{min_len}")
    print(f"Taxa de Erro de Caracteres: {taxa_erro_char:.2%}")
    
    # 9. Teste com múltiplas SNRs
    print("\n" + "=" * 60)
    print("TESTE COM DIFERENTES SNRs")
    print("=" * 60)
    
    snr_values = [0, 5, 10, 15, 20, 25]
    ber_values = []
    ber_teorica_values = []
    
    for snr in snr_values:
        # Configura novo canal
        channel.set_snr_db(snr)
        
        # Reset métricas
        system.reset_metrics()
        
        # Executa teste de BER
        ber = system.run_ber_test(num_bits=10000)
        ber_values.append(ber)
        
        # BER teórica
        ber_t = channel.get_theoretical_ber('16QAM')
        ber_teorica_values.append(ber_t)
        
        print(f"SNR = {snr:2d} dB | BER = {ber:.2e} | BER Teórica = {ber_t:.2e}")
    
    # 10. Plotagem de resultados
    print("\n" + "=" * 60)
    print("GERANDO GRÁFICOS")
    print("=" * 60)
    
    # Gráfico 1: BER vs SNR
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(snr_values, ber_values, 'bo-', label='BER Simulada', linewidth=2, markersize=8)
    plt.semilogy(snr_values, ber_teorica_values, 'r--', label='BER Teórica', linewidth=2)
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('Desempenho 16-QAM em Canal AWGN', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xlim([min(snr_values), max(snr_values)])
    
    # Gráfico 2: Diagrama de Constelação
    plt.subplot(1, 2, 2)
    
    # Transmite bits aleatórios para visualização
    test_bits = np.random.randint(0, 2, 1000)
    tx_signal = transmitter.modulate(test_bits)
    
    # Configura canal para SNR moderada
    channel.set_snr_db(15)
    rx_signal = channel.transmit(tx_signal)
    
    # Amostra símbolos
    baseband = receiver.downconvert(rx_signal)
    baseband = receiver.apply_agc(baseband)
    symbols = baseband[::transmitter.samples_per_symbol][:250]
    
    # Plota símbolos recebidos
    plt.scatter(np.real(symbols), np.imag(symbols), 
               s=30, c='blue', alpha=0.5, label='Símbolos Recebidos')
    
    # Plota constelação ideal
    plt.scatter(np.real(transmitter.constellation), np.imag(transmitter.constellation),
               s=200, c='red', marker='x', linewidths=3, label='Constelação Ideal')
    
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.xlabel('In-Phase (I)', fontsize=12)
    plt.ylabel('Quadrature (Q)', fontsize=12)
    plt.title(f'Diagrama de Constelação (SNR = 15 dB)', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSimulação concluída!")
    print("=" * 60)


if __name__ == "__main__":
    # Define seed para reprodutibilidade
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulação interrompida pelo usuário.")
    except Exception as e:
        print(f"\n\nErro durante a simulação: {e}")
        import traceback
        traceback.print_exc()
