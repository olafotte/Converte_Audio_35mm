import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
import scipy.io.wavfile as wav
from pathlib import Path
from PIL import Image
from scipy.signal import correlate, butter, lfilter

class ArqueologiaSonora:
    def __init__(self, pasta_imagens, config_json="config_rois.json", ratio=4.0):
        self.pasta = Path(pasta_imagens)
        self.ratio = ratio
        self.fps = 24
        with open(config_json, 'r') as f:
            self.config = json.load(f)
        self.arquivos = sorted([f for f in self.pasta.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
        
    def preparar_imagem(self, caminho):
        """Prepara a imagem rotacionando e espelhando conforme o Scanner V4."""
        with Image.open(caminho) as img:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.rotate(-90, resample=Image.BICUBIC, expand=True)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def extrair_sinal_full(self, img_cv, margem=150):
        """Extrai o sinal com correção polinomial e filtro passa-alta."""
        xa, ya, wa, ha = [int(c * self.ratio) for c in self.config['roi_audio']]
        xg, yg, wg, hg = [int(c * self.ratio) for c in self.config['roi_global']]
        
        img_global = img_cv[yg:yg+hg, xg:xg+wg]
        ya_exp = max(0, ya - margem)
        ha_exp = min(img_global.shape[0] - ya_exp, ha + (2 * margem))
        
        roi_gray = cv2.cvtColor(img_global[ya_exp:ya_exp+ha_exp, xa:xa+wa], cv2.COLOR_BGR2GRAY)
        sinal_bruto = np.mean(roi_gray.astype(np.float32) / 255.0, axis=1)
        
        # Correção Polinomial (Vignetting)
        t = np.arange(len(sinal_bruto))
        coeffs = np.polyfit(t, sinal_bruto, 2)
        sinal_plano = sinal_bruto - np.polyval(coeffs, t)
        
        # Filtro Passa-Alta (60Hz)
        nyq = 0.5 * (ha * self.fps)
        b, a = butter(4, 60 / nyq, btype='high')
        sinal_limpo = lfilter(b, a, sinal_plano)
        
        return sinal_limpo, roi_gray, ha

    def analisar_pipeline_frame(self, index=0):
        """Gera a visualização pedagógica de como a imagem vira som."""
        img_cv = self.preparar_imagem(self.arquivos[index])
        sinal, roi_img, _ = self.extrair_sinal_full(img_cv)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB))
        plt.title("1. ROI da Trilha Sonora (Imagem Expandida)")
        
        plt.subplot(3, 1, 2)
        plt.plot(sinal, color='orange')
        plt.title("2. Sinal Corrigido (Polinomial + High-pass)")
        
        plt.subplot(3, 1, 3)
        plt.plot(sinal[150:-150], color='blue') # Remove as margens para o gráfico final
        plt.fill_between(range(len(sinal[150:-150])), sinal[150:-150], color='blue', alpha=0.3)
        plt.title("3. Onda Sonora Final (Janela de 1/24s)")
        plt.tight_layout()
        plt.savefig("pipeline_analise.png")
        print("✅ Pipeline pedagógico salvo como pipeline_analise.png")

    def compilar_audio(self, limite_frames=240):
        """Processa a sequência, aplica phase matching e gera o áudio final."""
        audio_full = []
        viz_data = []
        margem = 150
        total = min(len(self.arquivos), limite_frames)
        
        print(f"🎙️ Compilando {total} frames...")
        
        for i in range(total):
            img_cv = self.preparar_imagem(self.arquivos[i])
            sinal_exp, roi_img, ha_samples = self.extrair_sinal_full(img_cv, margem)
            
            if i == 0:
                start_idx = margem
            else:
                # Phase Matching (Correlação de cauda e cabeça)
                ref = np.concatenate(audio_full[-2:])[-300:] # Referência do acumulado
                corr = correlate(ref, sinal_exp[:600], mode='full')
                match_idx = np.argmax(corr) - 299
                start_idx = max(0, min(match_idx, len(sinal_exp) - ha_samples))
            
            fatia = sinal_exp[start_idx : start_idx + ha_samples]
            audio_full.append(fatia)
            
            if i < 24: # Dados para o dashboard técnico
                viz_data.append({'roi': cv2.rotate(roi_img, cv2.ROTATE_90_CLOCKWISE), 
                                 'start': start_idx, 'sinal': fatia})
                
        # Exportação WAV
        audio_final = np.concatenate(audio_full)
        sr = ha_samples * self.fps
        audio_norm = (audio_final / (np.max(np.abs(audio_final)) + 1e-9) * 32767).astype(np.int16)
        wav.write("SOUNDTRACK_MASTER.wav", sr, audio_norm)
        print(f"✅ Áudio Master gerado ({sr}Hz)")
        
        self.gerar_dashboard(viz_data, ha_samples)

    def gerar_dashboard(self, viz_data, ha_samples):
        """Gera o painel mestre com cascata e forma de onda em fundo preto."""
        h_roi, w_roi = viz_data[0]['roi'].shape
        passo_y = int(h_roi * 0.25)
        canvas = np.zeros((24 * passo_y + h_roi, ha_samples * 24 + 200, 3), dtype=np.uint8) + 25
        cores_cmap = matplotlib.colormaps['gist_rainbow'].resampled(24)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [2, 1]})
        
        for i, d in enumerate(viz_data):
            y, x = i * passo_y, i * ha_samples
            roi_bgr = cv2.cvtColor(d['roi'], cv2.COLOR_GRAY2BGR)
            corte_img = roi_bgr[:, d['start'] : d['start'] + ha_samples]
            cor_rgba = cores_cmap(i)
            cor_bgr = (int(cor_rgba[2]*255), int(cor_rgba[1]*255), int(cor_rgba[0]*255))
            cv2.rectangle(corte_img, (0,0), (corte_img.shape[1]-1, corte_img.shape[0]-1), cor_bgr, 2)
            canvas[y : y+h_roi, x : x+ha_samples] = corte_img
            
            t = np.arange(i * ha_samples, (i + 1) * ha_samples)
            ax2.plot(t, d['sinal'], color=cor_rgba, linewidth=0.8)

        ax1.imshow(canvas, aspect='auto')
        ax1.set_title("Arqueologia Sonora: Cascata de ROIs")
        ax1.axis('off')
        
        ax2.set_facecolor('black')
        ax2.set_title("Sinal Consolidado (Phase Matched + Polinomial)", color='white')
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig("dashboard_tecnico.png", facecolor='#1e1e1e')
        print("✅ Dashboard salvo como dashboard_tecnico.png")
        plt.show()

def criar_visualizacao_cascata_mega_overlap(pasta_imagens="14042026"):
    scanner = ArqueologiaSonora(pasta_imagens)
    scanner.analisar_pipeline_frame(0) # Etapa 1 e 2 integrada
    scanner.compilar_audio(limite_frames=240) # Etapa 3 e 4 integrada

if __name__ == "__main__":
    criar_visualizacao_cascata_mega_overlap("14042026")