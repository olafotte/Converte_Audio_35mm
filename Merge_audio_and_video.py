import cv2
import os
import json
import datetime
import numpy as np
from pathlib import Path
from PIL import Image
import subprocess
from tqdm import tqdm
import librosa
import noisereduce as nr
import soundfile as sf
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, windows, correlate
from concurrent.futures import ThreadPoolExecutor
from setup_rois import criar_config_rois
from tracking import criar_tracking_geometrico
from detect_transition import detectar_saltos_com_grafico
from analizador_de_frequencias import analisar_ruido    




def extract_audio_v5(img_cv, ya, ha, xa, wa, margem=0):
    """
    Extrai áudio baseado na ROI do JSON com uma margem de segurança 
    de 5 pixels para permitir o casamento de fase (Phase Matching).
    """
    # Define os limites verticais com a margem de 5 pixels
    y_inicio = max(0, ya - margem)
    y_fim = min(img_cv.shape[0], ya + ha + margem)
    
    # Recorta a região da trilha sonora
    track_roi = cv2.cvtColor(img_cv[y_inicio:y_fim, xa:xa+wa], cv2.COLOR_BGR2GRAY)
    
    # Média Espacial para reduzir o ruído de "ventilador" (hiss)
    sig = np.mean(track_roi.astype(np.float32) / 255.0, axis=1)
    
    # Remove o offset DC para centralizar a onda em zero
    return sig - np.mean(sig)

def fundir_audio_fase(audio_acumulado, novo_chunk, search_range=150):
    """
    Alinhamento de Fase (Phase Matching) corrigido para evitar erros de broadcast.
    """
    if len(audio_acumulado) == 0:
        return novo_chunk

    # Comparamos o final do que já temos com o início do novo
    lookback = min(len(audio_acumulado), search_range)
    cauda = audio_acumulado[-lookback:]
    cabeca = novo_chunk[:search_range]
    
    # Correlação cruzada para achar o deslocamento ideal
    corr = correlate(cauda, cabeca, mode='full')
    offset_ideal = np.argmax(corr) - (len(cabeca) - 1)
    
    # Ponto onde o novo frame deve "entrar"
    ponto_insercao = len(audio_acumulado) - lookback + offset_ideal
    
    # Crossfade suave na emenda
    fade_len = 20
    
    if ponto_insercao > fade_len:
        # Prepara o áudio final com o tamanho correto do chunk atual
        novo_tamanho = ponto_insercao + len(novo_chunk)
        resultado = np.zeros(novo_tamanho)
        
        # 1. Copia o áudio antigo até o ponto de inserção
        resultado[:ponto_insercao] = audio_acumulado[:ponto_insercao]
        
        # 2. Aplica o Crossfade no ponto de encontro
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        
        resultado[ponto_insercao-fade_len:ponto_insercao] = (
            audio_acumulado[ponto_insercao-fade_len:ponto_insercao] * fade_out +
            novo_chunk[:fade_len] * fade_in
        )
        
        # 3. Copia o restante do novo chunk (Ajuste dinâmico de tamanho)
        resto_do_chunk = novo_chunk[fade_len:]
        resultado[ponto_insercao : ponto_insercao + len(resto_do_chunk)] = resto_do_chunk
        
        return resultado
    else:
        # Caso não haja espaço para crossfade, apenas concatena
        return np.concatenate([audio_acumulado, novo_chunk])
    

# --- FUNÇÕES AUXILIARES ---

def preparar_imagem(caminho, angulo):
    with Image.open(caminho) as img:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(angulo, resample=Image.BICUBIC, expand=True)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def restaurar_audio_final(input_wav, output_wav):
    # Carrega o áudio gerado
    y, sr = librosa.load(input_wav, sr=None)
    
    # Se o áudio for vazio ou muito curto, evitamos o processamento
    if len(y) < 100:
        print("⚠️ Áudio muito curto para restauração. Copiando original...")
        sf.write(output_wav, y, sr)
        return

    try:
        # Reduzimos o ruído com suavização zero para aceitar arquivos curtos
        y_denoised = nr.reduce_noise(
            y=y, 
            sr=sr, 
            prop_decrease=0.7, 
        )
    except Exception as e:
        print(f"⚠️ Erro no Denoise: {e}. Prosseguindo apenas com filtros.")
        y_denoised = y
    
    # Filtro passa-alta em 90Hz
    b, a = butter(5, 90 / (0.5 * sr), btype='high')
    y_filtered = lfilter(b, a, y_denoised)
    
    # Normaliza e salva
    sf.write(output_wav, librosa.util.normalize(y_filtered), sr)

# --- EXECUÇÃO ---

def criar_filme_com_audio(JSON_ROIS="config_rois.json", JSON_REFINADO="tracking_refined.json", PASTA_IMAGENS=Path("14042026"), FFMPEG_PATH=r"C:\ProgramData\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe", FPS=24, VISUALIZAR_DEBUG=False,LIMITE_TESTE_SEGUNDOS=0,SCALE_FACTOR=0.25):
    print("\n" + "="*60 + "\n      SCANNER 35MM V4 - PHASE MATCHING EDITION\n" + "="*60)
    
    arquivos = sorted([PASTA_IMAGENS / f for f in os.listdir(PASTA_IMAGENS) if f.lower().endswith(('.jpg', '.png'))])

    is_teste = LIMITE_TESTE_SEGUNDOS > 0
    if is_teste:
        print(f"⚠️ MODO TESTE ATIVO: Processando apenas {LIMITE_TESTE_SEGUNDOS} segundos.")
        arquivos = arquivos[:int(FPS * LIMITE_TESTE_SEGUNDOS)]

    if not os.path.exists(JSON_ROIS):
        criar_config_rois()

    config = json.load(open(JSON_ROIS))

    if not os.path.exists(JSON_REFINADO):
        criar_tracking_geometrico()

    mapa_refinado = json.load(open(JSON_REFINADO))
    ratio = 1/SCALE_FACTOR # Ajuste conforme seu preview (1 / 0.25)

    # ROIs Escaladas
    xg_f, yg_f, wg_f, hg_f = [int(c*ratio) for c in config['roi_global']]
    xa_f, ya_f, wa_f, ha_f = [int(c*ratio) for c in config['roi_audio']] # Ignoramos YA e HA para usar altura máxima
    xc_f, yc_f, wc_f, hc_f = [int(c*ratio) for c in config['roi_corte']]
    wc_f, hc_f = wc_f - (wc_f%2), hc_f - (hc_f%2)

    pipe = None

    cmd_ffmpeg = [FFMPEG_PATH, '-y', '-f', 'rawvideo', '-s', f'{wc_f}x{hc_f}', '-pix_fmt', 'bgr24', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p', 'temp_v.mp4']
    pipe = subprocess.Popen(cmd_ffmpeg, stdin=subprocess.PIPE)

    def worker_frame(f_path):
        img_h = preparar_imagem(f_path, -90)
        img_u = img_h[yg_f:yg_f+hg_f, xg_f:xg_f+wg_f]
        t = mapa_refinado[f_path.name]
        
        M = cv2.getRotationMatrix2D((img_u.shape[1]//2, img_u.shape[0]//2), -t['da'], 1.0)
        M[0,2] -= t['dx']*1
        M[1,2] -= t['dy']*1
        img_stab = cv2.warpAffine(img_u, M, (img_u.shape[1], img_u.shape[0]), flags=cv2.INTER_LANCZOS4)

        # --- VISUALIZAÇÃO DA TRILHA DE ÁUDIO ---
        if VISUALIZAR_DEBUG:
            # Desenha um retângulo VERDE na área exata do áudio (xa_f, ya_f, wa_f, ha_f)
            # Usamos uma espessura de 5 para ser bem visível
            cv2.rectangle(img_stab, (xa_f, ya_f), (xa_f + wa_f, ya_f + ha_f), (0, 255, 0), 5)
            
            # Opcional: Desenhar também a ROI de Corte final em AZUL
            cv2.rectangle(img_stab, (xc_f, yc_f), (xc_f + wc_f, yc_f + hc_f), (255, 0, 0), 3)

        audio_chunk = extract_audio_v5(img_stab, ya_f, ha_f, xa_f, wa_f, margem=1)        
        
        # Se você quiser que o box apareça no vídeo final, 
        # o desenho deve ser feito antes desta linha de recorte:
        frame_final = img_stab[yc_f:yc_f+hc_f, xc_f:xc_f+wc_f]

        return frame_final.tobytes(), audio_chunk, img_stab # Retornamos img_stab para o preview

    audio_list = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Note que agora o worker retorna 3 valores
        for f_bytes, a_chunk, img_debug in tqdm(executor.map(worker_frame, arquivos), total=len(arquivos), desc="Processando"):
            if pipe: pipe.stdin.write(f_bytes)
            audio_list.append(a_chunk)
            
            # Mostra a imagem na tela se o debug estiver ativo
            if VISUALIZAR_DEBUG:
                # Redimensiona apenas para o preview não ocupar a tela toda
                preview_img = cv2.resize(img_debug, (0,0), fx=0.4, fy=0.4)
                cv2.imshow("Preview Tracking & Audio ROI", preview_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                        
        cv2.destroyAllWindows()


    if pipe:
        pipe.stdin.close()
        pipe.wait()

    # Fusão Inteligente com Phase Matching
    audio_final = np.array([])
    for chunk in tqdm(audio_list, desc="Fundindo áudio (Fase)"):
        audio_final = fundir_audio_fase(audio_final, chunk)

    sr_real = int(len(audio_final) / (len(arquivos)/FPS))
    wav.write("audio_raw.wav", sr_real, (audio_final / np.max(np.abs(audio_final)) * 32767).astype(np.int16))
    restaurar_audio_final("audio_raw.wav", "audio_clean.wav")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    movie_name = f"MASTER_FINAL_HD_{timestamp}.mp4"

    subprocess.run([FFMPEG_PATH, '-y', '-i', 'temp_v.mp4', '-i', 'audio_clean.wav', '-c:a', 'aac', movie_name])
    print("\nPROCESSO CONCLUÍDO!")

if __name__ == "__main__":
    ANGULO = -90     # Ajuste conforme necessário para a orientação correta das suas imagens
    ESPELHAR = True  # Ative se suas imagens precisam ser espelhadas (dependendo de como foram escaneadas)
    SCALE_FACTOR = 0.25 # Ajuste para o preview (25% do tamanho original)
    JSON_OUTPUT="config_rois.json"
    JSON_REFINADO="tracking_refined.json" 
    PASTA_IMAGENS=Path("14042026") # Ajuste para o nome da pasta onde estão suas imagens
    LIMITE_TESTE_SEGUNDOS = 0 
    FFMPEG_PATH = r"C:\ProgramData\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe" # Ajuste para o caminho do seu ffmpeg.exe
    FPS = 24 # Taxa de quadros do vídeo final (ajuste conforme necessário)
    VISUALIZAR_DEBUG=False # Ative para ver as ROIs desenhadas e o preview durante o processamento (pode deixar mais lento)
    
    criar_config_rois(ANGULO, ESPELHAR, SCALE_FACTOR, JSON_OUTPUT, PASTA_IMAGENS)  # Garante que o JSON de ROIs exista antes de rodar o main
    criar_tracking_geometrico(ANGULO, ESPELHAR, SCALE_FACTOR, JSON_REFINADO, PASTA_IMAGENS) # Garante que o JSON de tracking refinado exista antes de rodar o main
    criar_filme_com_audio(JSON_OUTPUT, JSON_REFINADO, PASTA_IMAGENS, FFMPEG_PATH, FPS, VISUALIZAR_DEBUG,LIMITE_TESTE_SEGUNDOS,SCALE_FACTOR) # Roda o processo completo de criação do filme com áudio
    #detectar_saltos_com_grafico() # Roda a análise de transições para verificar estabilidade do vídeo
    analisar_ruido("audio_clean.wav") # Roda a análise de frequência para verificar o ruído do áudio final