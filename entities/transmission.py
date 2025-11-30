"""
Classe para simular transmissão completa de dados.
Integra transmissor, canal e receptor para análise de desempenho.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_transmitter import BaseTransmitter
from .base_receiver import BaseReceiver
from .base_channel import BaseChannel


class Transmission:
    """
    Simulador completo de transmissão digital.
    
    Integra um transmissor, um canal e um receptor para:
    - Simular transmissão de dados (texto, imagem, voz, vídeo)
    - Calcular métricas de desempenho (BER, SNR, throughput)
    - Analisar efeitos de diferentes canais
    """
    
    def __init__(self, transmitter: BaseTransmitter, channel: BaseChannel, 
                 receiver: BaseReceiver):
        """
        Inicializa a simulação de transmissão.
        
        Args:
            transmitter: Instância de transmissor
            channel: Instância de canal
            receiver: Instância de receptor
        """
        self.transmitter = transmitter
        self.channel = channel
        self.receiver = receiver
        
        # Validação básica
        if transmitter.sample_rate != receiver.sample_rate:
            raise ValueError("Taxa de amostragem do transmissor e receptor devem ser iguais")
        
        # Métricas
        self.metrics = {
            'ber': None,
            'ser': None,  # Symbol Error Rate
            'snr_db': None,
            'throughput': None,
            'total_bits': 0,
            'error_bits': 0,
            'total_symbols': 0,
            'error_symbols': 0,
            'transmission_time': None,  # Tempo de transmissão em segundos
            'bandwidth_used': None,     # Largura de banda utilizada em Hz
            'signal_duration': None     # Duração do sinal em segundos
        }
        
    def transmit_bits(self, bits: np.ndarray, calculate_metrics: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transmite bits através do sistema completo.
        
        Args:
            bits: Array de bits a transmitir
            calculate_metrics: Se True, calcula métricas de desempenho
            
        Returns:
            Tupla (bits_recebidos, sinal_recebido)
        """
        # 1. Modulação
        tx_signal = self.transmitter.modulate(bits)
        
        # Calcula tempo de transmissão e largura de banda
        signal_duration = len(tx_signal) / self.transmitter.sample_rate
        
        # Largura de banda utilizada (aproximação pela taxa de símbolos)
        if hasattr(self.transmitter, 'symbol_rate'):
            bandwidth_used = self.transmitter.symbol_rate
        else:
            # Fallback: usa taxa de amostragem como referência
            bandwidth_used = self.transmitter.sample_rate / self.transmitter.samples_per_symbol
        
        # Atualiza métricas
        self.metrics['signal_duration'] = signal_duration
        self.metrics['bandwidth_used'] = bandwidth_used
        self.metrics['transmission_time'] = signal_duration
        
        # 2. Transmissão pelo canal
        rx_signal = self.channel.transmit(tx_signal)
        
        # 3. Demodulação
        rx_bits = self.receiver.demodulate(rx_signal)
        
        # Ajusta comprimento (pode ter padding)
        min_len = min(len(bits), len(rx_bits))
        rx_bits = rx_bits[:min_len]
        tx_bits = bits[:min_len]
        
        # 4. Calcula métricas
        if calculate_metrics:
            self._calculate_ber(tx_bits, rx_bits)
        
        return rx_bits, rx_signal
    
    def transmit_text(self, text: str, encoding: str = 'utf-8') -> Tuple[str, Dict[str, Any]]:
        """
        Transmite texto através do sistema.
        
        Args:
            text: Texto a transmitir
            encoding: Codificação de caracteres
            
        Returns:
            Tupla (texto_recebido, métricas)
        """
        # Converte texto para bits
        data_bytes = text.encode(encoding)
        bits = self.transmitter.bits_from_bytes(data_bytes)
        
        # Transmite
        rx_bits, rx_signal = self.transmit_bits(bits)
        
        # Decodifica
        rx_bytes = self.receiver.bytes_from_bits(rx_bits)
        
        try:
            rx_text = rx_bytes.decode(encoding, errors='ignore')
        except Exception as e:
            rx_text = f"[Erro: {e}]"
        
        return rx_text, self.get_metrics()
    
    def transmit_bytes(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Transmite dados binários através do sistema.
        
        Args:
            data: Dados em bytes
            
        Returns:
            Tupla (dados_recebidos, métricas)
        """
        # Converte para bits
        bits = self.transmitter.bits_from_bytes(data)
        
        # Transmite
        rx_bits, _ = self.transmit_bits(bits)
        
        # Converte de volta para bytes
        rx_data = self.receiver.bytes_from_bits(rx_bits)
        
        return rx_data, self.get_metrics()
    
    def _calculate_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray):
        """
        Calcula a taxa de erro de bits (BER).
        
        Args:
            tx_bits: Bits transmitidos
            rx_bits: Bits recebidos
        """
        # Conta erros
        bit_errors = np.sum(tx_bits != rx_bits)
        total_bits = len(tx_bits)
        
        # Atualiza métricas
        self.metrics['total_bits'] += total_bits
        self.metrics['error_bits'] += bit_errors
        
        # Calcula BER
        if self.metrics['total_bits'] > 0:
            self.metrics['ber'] = self.metrics['error_bits'] / self.metrics['total_bits']
        else:
            self.metrics['ber'] = 0.0
    
    def _calculate_ser(self, tx_symbols: np.ndarray, rx_symbols: np.ndarray):
        """
        Calcula a taxa de erro de símbolos (SER).
        
        Args:
            tx_symbols: Símbolos transmitidos
            rx_symbols: Símbolos recebidos
        """
        # Conta erros de símbolos
        symbol_errors = np.sum(tx_symbols != rx_symbols)
        total_symbols = len(tx_symbols)
        
        # Atualiza métricas
        self.metrics['total_symbols'] += total_symbols
        self.metrics['error_symbols'] += symbol_errors
        
        # Calcula SER
        if self.metrics['total_symbols'] > 0:
            self.metrics['ser'] = self.metrics['error_symbols'] / self.metrics['total_symbols']
        else:
            self.metrics['ser'] = 0.0
    
    def measure_snr(self, signal_duration: float = 0.1) -> float:
        """
        Mede a SNR do sistema transmitindo um sinal conhecido.
        
        Args:
            signal_duration: Duração do sinal de teste em segundos
            
        Returns:
            SNR medida em dB
        """
        # Gera bits aleatórios
        num_bits = int(self.transmitter.sample_rate * signal_duration / 
                       self.transmitter.samples_per_symbol)
        
        if hasattr(self.transmitter, 'get_spectral_efficiency'):
            num_bits = int(num_bits * self.transmitter.get_spectral_efficiency())
        
        test_bits = np.random.randint(0, 2, num_bits)
        
        # Modula
        tx_signal = self.transmitter.modulate(test_bits)
        
        # Passa pelo canal
        rx_signal = self.channel.transmit(tx_signal)
        
        # Calcula SNR
        signal_power = np.mean(np.abs(tx_signal)**2)
        noise = rx_signal - tx_signal
        noise_power = np.mean(np.abs(noise)**2)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
        else:
            snr_db = float('inf')
        
        self.metrics['snr_db'] = snr_db
        return snr_db
    
    def calculate_throughput(self, duration: float = 1.0) -> float:
        """
        Calcula o throughput efetivo do sistema.
        
        Args:
            duration: Duração da medição em segundos
            
        Returns:
            Throughput em bits/segundo
        """
        # Calcula taxa de bits do transmissor
        if hasattr(self.transmitter, 'get_bit_rate'):
            bit_rate = self.transmitter.get_bit_rate()
        else:
            bit_rate = self.transmitter.symbol_rate
        
        # Throughput efetivo considera BER
        if self.metrics['ber'] is not None:
            effective_throughput = bit_rate * (1 - self.metrics['ber'])
        else:
            effective_throughput = bit_rate
        
        self.metrics['throughput'] = effective_throughput
        return effective_throughput
    
    def run_ber_test(self, num_bits: int = 10000) -> float:
        """
        Executa teste de BER com número específico de bits.
        
        Args:
            num_bits: Número de bits a transmitir
            
        Returns:
            BER medida
        """
        # Gera bits aleatórios
        test_bits = np.random.randint(0, 2, num_bits)
        
        # Transmite e calcula BER
        _, _ = self.transmit_bits(test_bits, calculate_metrics=True)
        
        return self.metrics['ber']
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna todas as métricas calculadas.
        
        Returns:
            Dicionário com métricas
        """
        return self.metrics.copy()
    
    def reset_metrics(self):
        """
        Reseta todas as métricas acumuladas.
        """
        self.metrics = {
            'ber': None,
            'ser': None,
            'snr_db': None,
            'throughput': None,
            'total_bits': 0,
            'error_bits': 0,
            'total_symbols': 0,
            'error_symbols': 0,
            'transmission_time': None,
            'bandwidth_used': None,
            'signal_duration': None
        }
    
    def get_system_info(self) -> str:
        """
        Retorna informações sobre o sistema de transmissão.
        
        Returns:
            String formatada com informações do sistema
        """
        info = "=== Sistema de Transmissão ===\n\n"
        info += f"Transmissor: {self.transmitter.get_modulation_type()}\n"
        info += f"  Taxa de Amostragem: {self.transmitter.sample_rate/1e6:.2f} MS/s\n"
        
        if hasattr(self.transmitter, 'symbol_rate'):
            info += f"  Taxa de Símbolos: {self.transmitter.symbol_rate/1e3:.2f} kbaud\n"
        
        if hasattr(self.transmitter, 'get_bit_rate'):
            info += f"  Taxa de Bits: {self.transmitter.get_bit_rate()/1e3:.2f} kbps\n"
        
        info += f"\nCanal: {self.channel.get_channel_type()}\n"
        
        for key, value in self.channel.get_parameters().items():
            if value is not None:
                info += f"  {key}: {value}\n"
        
        info += f"\nReceptor: {self.receiver.get_modulation_type()}\n"
        
        if self.metrics['ber'] is not None:
            info += "\n=== Métricas ===\n"
            info += f"BER: {self.metrics['ber']:.2e}\n"
            
            if self.metrics['snr_db'] is not None:
                info += f"SNR: {self.metrics['snr_db']:.2f} dB\n"
            
            if self.metrics['throughput'] is not None:
                info += f"Throughput: {self.metrics['throughput']/1e3:.2f} kbps\n"
            
            if self.metrics['transmission_time'] is not None:
                info += f"Tempo de Transmissão: {self.metrics['transmission_time']:.3f} s\n"
            
            if self.metrics['bandwidth_used'] is not None:
                info += f"Largura de Banda: {self.metrics['bandwidth_used']/1e3:.2f} kHz\n"
        
        return info
    
    def __str__(self) -> str:
        return self.get_system_info()
