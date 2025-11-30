# Digital Modulation Simulation System

A comprehensive Python-based digital communication system simulator for telecommunications courses, featuring multiple modulation schemes, channel models, and complete transmission pipeline.

## 📋 Overview

This project implements a modular framework for simulating digital communication systems with various modulation techniques, channel impairments, and performance analysis tools. Built with object-oriented design principles, it provides an educational platform for understanding modern telecommunications.

## 🏗️ Architecture

The system follows a three-layer architecture:

### Base Classes (Abstract)

- **`BaseTransmitter`**: Abstract transmitter with modulation interface, signal normalization, and carrier upconversion
- **`BaseReceiver`**: Abstract receiver with demodulation interface, AGC, and carrier downconversion  
- **`BaseChannel`**: Abstract channel for signal transmission and impairment modeling

### Concrete Implementations

#### Transmitters (`entities/`)

- **`BPSKTransmitter`**: Binary Phase-Shift Keying (1 bit/symbol)
  - Root Raised Cosine (RRC) pulse shaping
  - Phase modulation: 0° and 180°
  
- **`QPSKTransmitter`**: Quadrature Phase-Shift Keying (2 bits/symbol)
  - Four phase states: 45°, 135°, 225°, 315°
  - Gray coding for optimal BER
  
- **`ASK4Transmitter`**: 4-level Amplitude-Shift Keying (2 bits/symbol)
  - Four amplitude levels: -3, -1, +1, +3
  - Gray-coded constellation mapping
  
- **`QAM16Transmitter`**: 16-Quadrature Amplitude Modulation (4 bits/symbol)
  - 4×4 constellation with I/Q modulation
  - Gray coding and power normalization
  
- **`OFDMTransmitter`**: Orthogonal Frequency-Division Multiplexing (configurable)
  - Multiple orthogonal subcarriers (default: 128)
  - Cyclic prefix for ISI mitigation
  - Configurable subcarrier modulation (BPSK/QPSK/16-QAM)
  - IFFT-based efficient modulation
  - High spectral efficiency (>12k bits/s/Hz)

#### Receivers (`entities/`)

- **`BPSKReceiver`**: BPSK demodulation with zero-threshold decision
- **`ASK4Receiver`**: 4-ASK demodulation with threshold-based detection
- **`QAM16Receiver`**: 16-QAM demodulation with hard/soft decision
- **`OFDMReceiver`**: OFDM demodulation
  - FFT-based subcarrier recovery
  - Cyclic prefix removal
  - Per-subcarrier demodulation
  - EVM (Error Vector Magnitude) measurement

#### Channels (`entities/`)

- **`AWGNChannel`**: Additive White Gaussian Noise
  - Configurable SNR in dB
  - Theoretical BER calculation for BPSK/QPSK/16-QAM
  - Complex Gaussian noise generation

### Integration Layer

- **`Transmission`**: End-to-end transmission system
  - Integrates transmitter + channel + receiver
  - Automatic BER, SNR, throughput calculation
  - Transmission time and bandwidth metrics
  - Support for text, bytes, and arbitrary data

## 💻 Usage

### Basic Example

```python
from entities import QAM16Transmitter, QAM16Receiver, AWGNChannel, Transmission

# Create components
transmitter = QAM16Transmitter(sample_rate=1e6, carrier_freq=2.4e9)
receiver = QAM16Receiver(sample_rate=1e6, carrier_freq=2.4e9)
channel = AWGNChannel(snr_db=20)

# Integrate into transmission system
system = Transmission(transmitter, channel, receiver)

# Transmit text
message = "Hello, OFDM!"
rx_text, metrics = system.transmit_text(message)

print(f"Received: {rx_text}")
print(f"BER: {metrics['ber']:.2e}")
print(f"Transmission Time: {metrics['transmission_time']*1000:.2f} ms")
```

### OFDM Example

```python
from entities import OFDMTransmitter, OFDMReceiver, AWGNChannel, Transmission

# Create OFDM system with QPSK subcarrier modulation
tx = OFDMTransmitter(
    sample_rate=1e6,
    num_subcarriers=128,
    cp_length=32,
    subcarrier_modulation='QPSK',
    carrier_freq=2.4e9
)

rx = OFDMReceiver(
    sample_rate=1e6,
    num_subcarriers=128,
    cp_length=32,
    subcarrier_modulation='QPSK',
    carrier_freq=2.4e9
)

channel = AWGNChannel(snr_db=25)
system = Transmission(tx, channel, rx)

# Transmit audio data
audio_bytes = generate_audio()  # Your audio data
rx_audio, metrics = system.transmit_bytes(audio_bytes)

print(f"Bit Rate: {tx.get_bit_rate()/1e6:.2f} Mbps")
print(f"Spectral Efficiency: {tx.get_spectral_efficiency():.2f} bits/s/Hz")
```

## 🧪 Test Suite

The project includes comprehensive test scripts demonstrating various transmission scenarios:

### Single-Carrier Tests

- **`test_qam16_transmission.py`**: Text transmission with 16-QAM
  - 10,000 character text transmission
  - BER vs SNR curve analysis (0-25 dB)
  - Constellation diagram visualization
  - Carrier frequency: 450 MHz
  
- **`test_ask4_long_transmission.py`**: Long-distance 4-ASK transmission
  - 10k characters in <2 seconds
  - Eye diagram and temporal dispersion analysis
  - Histogram of received symbols

- **`test_qam16_image_transmission.py`**: RGB image transmission
  - 100×100 pixel RGB image
  - Transmission time: <3 seconds
  - PSNR (Peak Signal-to-Noise Ratio) measurement
  - Before/after comparison visualization

- **`test_qam16_video_transmission.py`**: Video transmission
  - 10 grayscale frames (100×100 pixels)
  - Transmission time: <1 second
  - Per-frame PSNR analysis
  - Animated sequence reconstruction

### Multi-Carrier Tests

- **`test_ofdm_audio_transmission.py`**: OFDM audio transmission
  - 5 seconds of audio @ 16 kHz, 16 bits/sample, mono
  - 160,000 bytes (1.28 Mbits) transmitted in 813 ms
  - Transmission rate: 1.575 Mbps
  - Comprehensive visualization suite:
    - Waveform comparison (original vs received)
    - Frequency spectrum analysis
    - Error signal distribution
    - OFDM constellation diagram
    - BER over time
    - THD (Total Harmonic Distortion) measurement
    - EVM RMS analysis

## 📊 Performance Metrics

All transmission systems automatically calculate:

- **BER** (Bit Error Rate): Ratio of bit errors to total bits
- **SER** (Symbol Error Rate): Ratio of symbol errors to total symbols
- **SNR** (Signal-to-Noise Ratio): In dB
- **Throughput**: Effective data rate in bps
- **Transmission Time**: Actual signal duration in seconds
- **Bandwidth Used**: Occupied spectrum in Hz
- **PSNR** (for images/video): Image quality metric in dB
- **EVM RMS** (for OFDM): Constellation error magnitude in %
- **THD** (for audio): Total harmonic distortion in %

## 🔧 Dependencies

```python
numpy          # Signal processing and numerical operations
matplotlib     # Visualization and plotting
Pillow (PIL)   # Image processing for transmission tests
```

Install with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```text
radio-frequency/
├── entities/
│   ├── __init__.py
│   ├── base_transmitter.py      # Abstract transmitter
│   ├── base_receiver.py         # Abstract receiver
│   ├── base_channel.py          # Abstract channel
│   ├── bpsk_transmitter.py      # BPSK implementation
│   ├── bpsk_receiver.py
│   ├── ask4_transmitter.py      # 4-ASK implementation
│   ├── ask4_receiver.py
│   ├── qam16_transmitter.py     # 16-QAM implementation
│   ├── qam16_receiver.py
│   ├── ofdm_transmitter.py      # OFDM implementation
│   ├── ofdm_receiver.py
│   ├── awgn_channel.py          # AWGN channel model
│   └── transmission.py          # Integration layer
├── test_qam16_transmission.py
├── test_ask4_long_transmission.py
├── test_qam16_image_transmission.py
├── test_qam16_video_transmission.py
├── test_ofdm_audio_transmission.py
├── requirements.txt
└── README.md
```

## 🎓 Educational Applications

This simulator is ideal for:

- Understanding digital modulation theory
- Analyzing BER performance under noise
- Comparing single-carrier vs multi-carrier systems
- Visualizing constellation diagrams and eye patterns
- Studying ISI mitigation with pulse shaping
- Exploring OFDM principles and cyclic prefix
- Measuring spectral efficiency trade-offs
- Real-world data transmission scenarios (text, images, audio, video)

## 🚀 Running Tests

Execute individual test scripts:

```bash
python test_qam16_transmission.py
python test_ofdm_audio_transmission.py
```

Each test generates comprehensive visualizations and performance metrics.

## 📈 Features

✅ Multiple modulation schemes (BPSK, 4-ASK, 16-QAM, OFDM)  
✅ Gray coding for optimal BER performance  
✅ RRC pulse shaping for ISI reduction  
✅ Automatic Gain Control (AGC)  
✅ Carrier frequency upconversion/downconversion  
✅ AWGN channel with configurable SNR  
✅ Comprehensive metrics (BER, SNR, PSNR, EVM, THD)  
✅ Real-world transmission tests (text, images, audio, video)  
✅ Rich visualization suite (constellations, spectra, eye diagrams)  
✅ OFDM with cyclic prefix and FFT/IFFT processing  

## 📝 License

See `LICENSE` file for details.

## 👥 Contributors

Developed for IFES Telecommunications Electronics course.
