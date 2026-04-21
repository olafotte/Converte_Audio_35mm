import cv2
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image


def preparar_imagem_setup(caminho, angulo, espelhar):
    """
    Aplica as transformações ANTES da seleção de ROIs.
    """
    with Image.open(caminho) as img:
        # 1. Espelhar
        if espelhar:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 2. Girar (expand=True é vital para não cortar a imagem)
        img = img.rotate(angulo, resample=Image.BICUBIC, expand=True)
        
        # Converte para OpenCV (BGR)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def criar_config_rois(ANGULO=-90, ESPELHAR=True, SCALE_FACTOR=0.25, JSON_OUTPUT="config_rois.json", PASTA_IMAGENS=Path("14042026")):
    # Pega a primeira imagem da pasta
    arquivos = sorted([PASTA_IMAGENS / f for f in os.listdir(PASTA_IMAGENS) if f.lower().endswith(('.jpg', '.png'))])
    if not arquivos:
        print("❌ Erro: Nenhuma imagem encontrada!")
        return

    print(f"🔄 Transformando imagem de referência (Giro: {ANGULO}°, Espelhar: {ESPELHAR})...")
    img_full = preparar_imagem_setup(arquivos[0], ANGULO, ESPELHAR)
    
    # Redimensiona para o preview (25%)
    width = int(img_full.shape[1] * SCALE_FACTOR)
    height = int(img_full.shape[0] * SCALE_FACTOR)
    img_preview = cv2.resize(img_full, (width, height), interpolation=cv2.INTER_AREA)

    print("\n--- SELEÇÃO DE REGIÕES (Pressione ENTER após cada seleção) ---")
    
    # 1. ROI Global
    print("1. Selecione a ROI GLOBAL (área de estabilização)...")
    roi_global = cv2.selectROI("1. ROI Global", img_preview, False)
    cv2.destroyWindow("1. ROI Global")

    # Recorta a imagem para as próximas seleções serem relativas à ROI Global
    xg, yg, wg, hg = [int(v) for v in roi_global]
    img_crop_global = img_preview[yg:yg+hg, xg:xg+wg]

    # 2. ROI Áudio (Relativa à Global)
    print("2. Selecione a ROI de ÁUDIO (dentro da área global)...")
    roi_audio = cv2.selectROI("2. ROI Audio", img_crop_global, False)
    cv2.destroyWindow("2. ROI Audio")

    # 3. ROI Corte (Relativa à Global)
    print("3. Selecione a ROI de CORTE (dentro da área global)...")
    roi_corte = cv2.selectROI("3. ROI Corte", img_crop_global, False)
    cv2.destroyWindow("3. ROI Corte")

    # Monta o dicionário (Salvamos os valores da escala 25% para bater com o ratio 4.0 do seu V4)
    config = {
        "roi_global": [int(v) for v in roi_global],
        "roi_audio": [int(v) for v in roi_audio],
        "roi_corte": [int(v) for v in roi_corte]
    }

    # Salva o JSON
    with open(JSON_OUTPUT, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n✅ Arquivo {JSON_OUTPUT} criado com sucesso!")
    print(f"As ROIs foram salvas na escala de {SCALE_FACTOR*100}%")

if __name__ == "__main__":

    criar_config_rois(ANGULO=-90, ESPELHAR=True, SCALE_FACTOR=0.25, JSON_OUTPUT="config_rois.json", PASTA_IMAGENS=Path("14042026"))