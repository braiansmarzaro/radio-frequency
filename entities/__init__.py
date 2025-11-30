"""
Módulo de entidades para simulação de modulação digital.

Contém implementações de transmissores, receptores e canais
para análise de sistemas de comunicação digital.
"""

from .base_transmitter import BaseTransmitter
from .bpsk_transmitter import BPSKTransmitter
from .qam16_transmitter import QAM16Transmitter
from .base_receiver import BaseReceiver
from .qam16_receiver import QAM16Receiver
from .base_channel import BaseChannel
from .awgn_channel import AWGNChannel
from .transmission import Transmission

__all__ = [
    'BaseTransmitter',
    'BPSKTransmitter',
    'QAM16Transmitter',
    'BaseReceiver',
    'QAM16Receiver',
    'BaseChannel',
    'AWGNChannel',
    'Transmission',
]

__version__ = '0.1.0'
