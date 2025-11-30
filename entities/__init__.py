"""
Módulo de entidades para simulação de modulação digital.

Contém implementações de transmissores, receptores e canais
para análise de sistemas de comunicação digital.
"""

from .base_transmitter import BaseTransmitter
from .bpsk_transmitter import BPSKTransmitter
from .qam16_transmitter import QAM16Transmitter
from .ask4_transmitter import ASK4Transmitter
from .base_receiver import BaseReceiver
from .bpsk_receiver import BPSKReceiver
from .qam16_receiver import QAM16Receiver
from .ask4_receiver import ASK4Receiver
from .base_channel import BaseChannel
from .awgn_channel import AWGNChannel
from .transmission import Transmission

__all__ = [
    'BaseTransmitter',
    'BPSKTransmitter',
    'QAM16Transmitter',
    'ASK4Transmitter',
    'BaseReceiver',
    'BPSKReceiver',
    'QAM16Receiver',
    'ASK4Receiver',
    'BaseChannel',
    'AWGNChannel',
    'Transmission',
]

__version__ = '0.1.0'
