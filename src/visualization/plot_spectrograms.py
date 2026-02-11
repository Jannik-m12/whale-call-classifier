def plot_spectrogram(audio_data, sample_rate, title='Spectrogram'):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import spectrogram

    # Generate the spectrogram
    frequencies, times, Sxx = spectrogram(audio_data, sample_rate)

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.tight_layout()
    plt.show()