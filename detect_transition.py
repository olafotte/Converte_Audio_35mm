import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURAÇÕES OTIMIZADAS ---
PASTA_IMAGENS = Path("14042026")
ANGULO = -90
ESPELHAR = True
LIMITE_TRANSICAO = 15.0  # Sensibilidade do corte
SCALE_PREVIEW = 0.3      # Tamanho da imagem na tela
TAMANHO_ANALISE = (320, 240) # Resolução reduzida para cálculo ultra rápido

def preparar_imagem_otimizada(caminho):
    """Lê e prepara a imagem de forma performática."""
    try:
        with Image.open(caminho) as img:
            if ESPELHAR:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # NEAREST é muito mais rápido que BICUBIC para fins de análise
            img = img.rotate(ANGULO, resample=Image.NEAREST, expand=True)
            
            frame_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # Redimensiona para o cálculo de diferença
            gray_small = cv2.resize(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY), TAMANHO_ANALISE)
            return gray_small, frame_cv
    except Exception as e:
        print(f"Erro ao processar {caminho.name}: {e}")
        return None, None

def detectar_saltos_com_grafico():
    arquivos = sorted([PASTA_IMAGENS / f for f in os.listdir(PASTA_IMAGENS) if f.lower().endswith(('.jpg', '.png'))])
    
    if len(arquivos) < 2:
        print("❌ Poucas imagens para comparar.")
        return

    total_frames = len(arquivos)
    # Define o intervalo de 1% para atualização do gráfico
    intervalo_update = max(1, total_frames // 100) 

    # --- SETUP DO GRÁFICO ---
    plt.ion() 
    fig, ax = plt.subplots(figsize=(10, 5))
    indices, scores = [], []
    
    line, = ax.plot([], [], 'b-', linewidth=1, label='Diferença (Score)')
    ax.axhline(y=LIMITE_TRANSICAO, color='r', linestyle='--', alpha=0.6, label='Limite')
    
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, 100)
    ax.set_title('Análise de Estabilidade (Atualização a cada 10%)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    suspicious_frames = []
    
    # Processamento com ThreadPool para carregar imagens em paralelo
    print(f"🔍 Analisando {total_frames} frames...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Carrega o primeiro frame para iniciar a comparação
        gray_anterior, _ = preparar_imagem_otimizada(arquivos[0])

        # Loop otimizado
        for i, (gray_atual, frame_full) in enumerate(tqdm(executor.map(preparar_imagem_otimizada, arquivos[1:]), total=total_frames-1), 1):
            if gray_atual is None: continue

            # 1. Cálculo do Score (Diferença Média)
            diff = cv2.absdiff(gray_anterior, gray_atual)
            score = np.mean(diff)
            
            indices.append(i)
            scores.append(score)

            # 2. Atualização Condicional do Gráfico (A cada 10%)
            if i % intervalo_update == 0 or i == total_frames - 1:
                line.set_data(indices, scores)
                
                # Ajuste de escala do eixo Y se necessário
                if score > ax.get_ylim()[1]:
                    ax.set_ylim(0, score + 10)
                
                fig.canvas.draw()
                fig.canvas.flush_events()

            # 3. Registro de Saltos e Preview Visual
            if score > LIMITE_TRANSICAO:
                suspicious_frames.append({"nome": arquivos[i].name, "score": score})
                
                # Só mostra o preview se houver um salto para economizar recursos
                preview = cv2.resize(frame_full, (0,0), fx=SCALE_PREVIEW, fy=SCALE_PREVIEW)
                cv2.putText(preview, f"SALTO: {score:.1f}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Analisador de Transicoes", preview)

            # O waitKey é necessário para manter a janela do OpenCV viva
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            gray_anterior = gray_atual

    # --- FINALIZAÇÃO ---
    plt.ioff() 
    print(f"\n✅ Concluído. {len(suspicious_frames)} transições detectadas.")
    plt.savefig("resultado_analise_estabilidade.png")
    
    cv2.destroyAllWindows()
    plt.show() 

if __name__ == "__main__":
    detectar_saltos_com_grafico()