import soundfile as sf
import scipy.signal
import scipy.fft
import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def analyze_audio(file_path, segment_length=10, sample_rate=1000):
    # Load the audio file
    audio, sr = sf.read(file_path)

    # Resample if necessary
    if sr != sample_rate:
        num_samples = int(len(audio) * sample_rate / sr)
        audio = scipy.signal.resample(audio, num_samples)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Extract the desired segment
    num_samples = segment_length * sample_rate
    audio_segment = audio[:num_samples]

    # Fourier Transform
    fft_audio = scipy.fft.fft(audio_segment)
    fft_freq = scipy.fft.fftfreq(len(audio_segment), 1/sample_rate)

    # Continuous Wavelet Transform (CWT)
    max_freq = sample_rate // 2
    widths = np.logspace(0, np.log10(max_freq), num=50)
    cwt_audio, _ = pywt.cwt(audio_segment, widths, 'mexh')

    # Normalize CWT for better visualization
    cwt_audio_abs_log = np.log1p(np.abs(cwt_audio))
    vmin, vmax = np.percentile(cwt_audio_abs_log, [5, 95])

    # Discrete Wavelet Transform (DWT)
    coeffs = pywt.wavedec(audio_segment, 'db1', level=4)
    dwt_audio = pywt.waverec(coeffs, 'db1')

    # Time vector for plotting
    t = np.linspace(0, len(audio_segment)/sample_rate, num=len(audio_segment))

    # Plotting
    plt.figure(figsize=(15, 10))

    # Original Audio Signal
    plt.subplot(4, 1, 1)
    plt.plot(t, audio_segment)
    plt.title('Original Audio Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Fourier Transform
    plt.subplot(4, 1, 2)
    plt.plot(fft_freq, np.abs(fft_audio))
    plt.title('Fourier Transform')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

    # Continuous Wavelet Transform
    plt.subplot(4, 1, 3)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(cwt_audio_abs_log, extent=[0, len(audio_segment)/sample_rate, 1, max_freq],
               cmap='viridis', aspect='auto', norm=norm)
    plt.title('Continuous Wavelet Transform - Log Scale')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # Discrete Wavelet Transform
    plt.subplot(4, 1, 4)
    plt.plot(t, dwt_audio)
    plt.title('Discrete Wavelet Transform (Reconstructed Signal)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


# Usage
file_path = './Guilty Hero.mp3'
analyze_audio(file_path, segment_length=10, sample_rate=1000)
