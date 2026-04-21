import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
         

# --- CONFIGURAÇÃO DINÂMICA ROBUSTA ---
PERCENTUAL_AREA = 99.0      # Usamos o valor que cobre 99.9% da imagem (ignora pixels isolados)
PERCENTUAL_THRESHOLD = 0.95 # O threshold será 95% desse valor robusto
MAX_DY_LIMITE = 50       

# Variáveis globais para interação
click_x, click_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y

def preparar_imagem_tracking(caminho, angulo, espelhar):
    with Image.open(caminho) as img:
        if espelhar:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(angulo, resample=Image.BICUBIC, expand=True)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encontrar_centro_furo(img_gray, x_alvo, y_alvo, threshold_dinamico):
    _, thresh = cv2.threshold(img_gray, threshold_dinamico, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    melhor_centro = None
    menor_distancia = float('inf')

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            dist = np.sqrt((cx - x_alvo)**2 + (cy - y_alvo)**2)
            if dist < menor_distancia and dist < 150: # Raio de busca um pouco maior
                menor_distancia = dist
                melhor_centro = (cx, cy)
                
    return melhor_centro, thresh

def criar_tracking_geometrico(ANGULO=-90, ESPELHAR=True, SCALE_FACTOR=0.25, JSON_REFINADO="tracking_refined.json", PASTA_IMAGENS=Path("14042026")):
    global click_x, click_y
    arquivos = sorted([f for f in PASTA_IMAGENS.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
    if not arquivos: return

    # --- INICIALIZAÇÃO ---
    frame_orig = preparar_imagem_tracking(arquivos[0], ANGULO, ESPELHAR)
    gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    mapa_refinado = {}

    # --- PASSO 1: CLIQUE NO FURO ---
    window_name = "PASSO 1: Clique no centro do furo branco"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    h, w = frame_orig.shape[:2]
    frame_display = cv2.resize(frame_orig, (int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)))

    print("\n👉 Clique no centro do furo branco...")
    while click_x == -1:
        cv2.imshow(window_name, frame_display)
        if cv2.waitKey(1) & 0xFF == 27: return 

    # Cálculo do Threshold Robusto para o Frame 0
    # np.percentile ignora os 0.1% de pixels mais brilhantes (ruído)
    valor_referencia_orig = np.percentile(gray_orig, PERCENTUAL_AREA)
    thresh_orig = int(valor_referencia_orig * PERCENTUAL_THRESHOLD)

    real_click_x, real_click_y = int(click_x / SCALE_FACTOR), int(click_y / SCALE_FACTOR)
    centro_ref, _ = encontrar_centro_furo(gray_orig, real_click_x, real_click_y, thresh_orig)

    if not centro_ref:
        print(f"❌ Erro inicial. Ref: {valor_referencia_orig:.1f} | Thresh: {thresh_orig}")
        cv2.destroyAllWindows()
        return

    cv2.destroyWindow(window_name)

    # --- PASSO 2: SELECIONAR ROI DE BUSCA ---
    print("\n👉 Desenhe a ROI de BUSCA...")
    roi_search = cv2.selectROI("PASSO 2: Selecione a regiao de busca", frame_display, False)
    cv2.destroyAllWindows()

    sx, sy, sw, sh = [int(v / SCALE_FACTOR) for v in roi_search]

    # --- PASSO 3: LOOP DE TRACKING ---
    print(f"\n🚀 Iniciando Tracking Robusto (Percentil {PERCENTUAL_AREA}%)...")
    window_preview = "Tracking com Filtro de Ruido (B&W)"
    
    for i, path in enumerate(tqdm(arquivos)):
        frame_atual = preparar_imagem_tracking(path, ANGULO, ESPELHAR)
        gray_atual = cv2.cvtColor(frame_atual, cv2.COLOR_BGR2GRAY)
        
        # 1. CÁLCULO DO BRANCO ROBUSTO (Ignora pixels isolados/ruído)
        # Pegamos o valor onde 99.9% dos pixels estão abaixo dele
        valor_branco_robusto = np.percentile(gray_atual, PERCENTUAL_AREA)
        
        # 2. THRESHOLD DINÂMICO
        thresh_dinamico = int(valor_branco_robusto * PERCENTUAL_THRESHOLD)
        
        # 3. ANALISAR ROI
        roi_gray = gray_atual[sy:sy+sh, sx:sx+sw]
        centro_relativo, _ = encontrar_centro_furo(roi_gray, sw//2, sh//2, thresh_dinamico)

        # 4. PREPARAÇÃO VISUAL
        _, frame_bw = cv2.threshold(gray_atual, thresh_dinamico, 255, cv2.THRESH_BINARY)
        preview_img = cv2.cvtColor(frame_bw, cv2.COLOR_GRAY2BGR)

        # ESCREVER VALORES NO CANTO SUPERIOR DIREITO
        # "Pico Real" vs "Branco Médio Robusto"
        max_abs = np.max(gray_atual)
        texto_debug = f"Pico: {max_abs} | Robusto: {int(valor_branco_robusto)}"
        cv2.putText(preview_img, texto_debug, (frame_atual.shape[1] - 850, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

        if i == 0:
            dx, dy = 0.0, 0.0
        elif centro_relativo:
            cx_atual = centro_relativo[0] + sx
            cy_atual = centro_relativo[1] + sy
            dx = float(cx_atual - centro_ref[0])
            dy = float(cy_atual - centro_ref[1])

            if abs(dy) > MAX_DY_LIMITE or abs(dx) > MAX_DY_LIMITE:
                dx, dy = 0.0, 0.0
                color = (0, 165, 255) # Laranja
            else:
                color = (0, 0, 255)   # Vermelho

            cv2.drawMarker(preview_img, (cx_atual, cy_atual), color, cv2.MARKER_CROSS, 40, 3)
            cv2.rectangle(preview_img, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
        else:
            dx, dy = 0.0, 0.0
            cv2.putText(preview_img, "PERDIDO", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        mapa_refinado[path.name] = {"dx": dx, "dy": dy, "da": 0.0}

        preview_resized = cv2.resize(preview_img, (int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)))
        cv2.imshow(window_preview, preview_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    with open(JSON_REFINADO, 'w', encoding='utf-8') as f:
        json.dump(mapa_refinado, f, indent=4)
    
    cv2.destroyAllWindows()
    print(f"\n✅ Concluído com filtragem de hot pixels.")

if __name__ == "__main__":

    criar_tracking_geometrico(ANGULO=-90, ESPELHAR=True, SCALE_FACTOR=0.25, JSON_REFINADO="tracking_refined.json", PASTA_IMAGENS=Path("14042026"))