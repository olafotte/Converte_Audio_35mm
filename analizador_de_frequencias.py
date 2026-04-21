import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def analisar_ruido(caminho_audio, output_plot="analise_frequencia.png"):
    # 1. Carregar o áudio
    y, sr = librosa.load(caminho_audio, sr=None)
    
    # 2. Calcular a FFT (Espectro de Frequência)
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:sr*5], n_fft=n_fft)) # Analisa os primeiros 5 segundos
    magnitude = np.mean(ft, axis=1)
    frequencias = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 3. Criar a visualização
    plt.figure(figsize=(12, 10))

    # Subplot 1: Espectro de Magnitude (Para achar picos específicos)
    plt.subplot(2, 1, 1)
    plt.plot(frequencias, librosa.amplitude_to_db(magnitude, ref=np.max))
    plt.title('Espectro de Magnitude (Procure por picos verticais)')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xscale('log') # Escala logarítmica ajuda a ver baixas frequências (hum)
    plt.xlim(20, sr//2)

    # Subplot 2: Espectrograma (Para ver se o ruído oscila)
    plt.subplot(2, 1, 2)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma (Linhas horizontais contínuas = ruído constante)')

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Análise salva em: {output_plot}")

if __name__ == "__main__":
    # Substitua pelo nome do seu arquivo gerado
    analisar_ruido("audio_clean.wav")