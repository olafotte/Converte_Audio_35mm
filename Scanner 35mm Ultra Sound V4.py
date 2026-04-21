import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading

# Importando suas funções originais
from Merge_audio_and_video import criar_filme_com_audio
from setup_rois import criar_config_rois
from tracking import criar_tracking_geometrico
from analizador_de_frequencias import analisar_ruido
from detect_transition import detectar_saltos_com_grafico
from Sound_generation import ArqueologiaSonora

class ScannerInterface(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Scanner 35mm Ultra Sound - Painel de Controle")
        self.geometry("600x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- VARIÁVEIS ---
        self.path_imagens = ctk.StringVar(value="Selecione a pasta...")
        self.ffmpeg_path = ctk.StringVar(value=r"C:\ProgramData\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe")
        self.fps = ctk.StringVar(value="24")
        self.angulo = ctk.DoubleVar(value=-90)
        self.espelhar = ctk.BooleanVar(value=True)
        self.scale_factor = ctk.DoubleVar(value=0.25)
        self.limite_teste = ctk.IntVar(value=0) # 0 significa filme completo

        self.setup_ui()

    def setup_ui(self):
        # Título
        lbl_title = ctk.CTkLabel(self, text="SCANNER 35MM V4", font=("Roboto", 24, "bold"))
        lbl_title.pack(pady=20)

        # Seleção de Pasta
        frame_pasta = ctk.CTkFrame(self)
        frame_pasta.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(frame_pasta, text="Pasta das Imagens:").pack(side="left", padx=10)
        ctk.CTkEntry(frame_pasta, textvariable=self.path_imagens, width=300).pack(side="left", padx=10)
        ctk.CTkButton(frame_pasta, text="Abrir", width=50, command=self.selecionar_pasta).pack(side="left")

        # Configurações de Imagem
        frame_config = ctk.CTkFrame(self)
        frame_config.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(frame_config, text="Ângulo:").grid(row=0, column=0, padx=10, pady=5)
        ctk.CTkEntry(frame_config, textvariable=self.angulo, width=60).grid(row=0, column=1)

        ctk.CTkLabel(frame_config, text="Scale Factor:").grid(row=0, column=2, padx=10, pady=5)
        ctk.CTkEntry(frame_config, textvariable=self.scale_factor).grid(row=0, column=3)

        ctk.CTkCheckBox(frame_config, text="Espelhar Imagem", variable=self.espelhar).grid(row=0, column=4, padx=20)

        ctk.CTkLabel(frame_config, text="FPS:").grid(row=2, column=0, padx=10, pady=5)
        ctk.CTkEntry(frame_config, textvariable=self.fps, width=60).grid(row=2, column=1)
        ctk.CTkLabel(frame_config, text="Limite Teste (s):").grid(row=2, column=2, padx=10)
        ctk.CTkEntry(frame_config, textvariable=self.limite_teste, width=60).grid(row=2, column=3)

        # O texto de ajuda (Label pequeno e cinza)
        lbl_ajuda = ctk.CTkLabel(frame_config, text="(0 para completo)", font=("Roboto", 10), text_color="gray")
        lbl_ajuda.grid(row=2, column=4, padx=5)

        # FFMPEG Path
        frame_ffmpeg = ctk.CTkFrame(self)
        frame_ffmpeg.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(frame_ffmpeg, text="Caminho FFMPEG:").pack(side="left", padx=10)
        ctk.CTkEntry(frame_ffmpeg, textvariable=self.ffmpeg_path, width=350).pack(side="left")

        # Botões de Ação

        self.btn_transitions = ctk.CTkButton(self, text="1. DETECTAR TRANSIÇÕES", height=50, fg_color="green",font=("Roboto", 16, "bold"),hover_color="#014d11", command=self.run_transition_detection)
        self.btn_transitions.pack(pady=10, padx=20, fill="x")

        self.btn_rois = ctk.CTkButton(self, text="2. Configurar ROIs (Visual)", height=50, fg_color="green", font=("Roboto", 16, "bold"), hover_color="#014d11", command=self.run_setup_rois)
        self.btn_rois.pack(pady=10, padx=20, fill="x")

        self.btn_track = ctk.CTkButton(self, text="3. Iniciar Tracking", height=50,  fg_color="green", font=("Roboto", 16, "bold"), hover_color="#014d11", command=self.run_tracking)
        self.btn_track.pack(pady=10, padx=20, fill="x")

        self.btn_render = ctk.CTkButton(self, text="4. RENDERIZAR FILME FINAL", height=50, font=("Roboto", 16, "bold"), command=self.run_main_process)
        self.btn_render.pack(pady=10, padx=20, fill="x")

        self.btn_audio_freq = ctk.CTkButton(self, text="5. ANALISAR FREQUÊNCIAS DE ÁUDIO", height=50, font=("Roboto", 16, "bold"), command=self.run_audio_frequency_analysis)
        self.btn_audio_freq.pack(pady=10, padx=20, fill="x")

        self.btn_arqueologia = ctk.CTkButton(self, text="6. ARQUEOLOGIA SONORA (Pipeline Visual)", height=50, font=("Roboto", 16, "bold"), command=self.criar_visualizacao_cascata_mega_overlap)
        self.btn_arqueologia.pack(pady=10, padx=20, fill="x")

        # Status
        self.status_label = ctk.CTkLabel(self, text="Status: Aguardando...", text_color="yellow")
        self.status_label.pack(pady=10)

    def selecionar_pasta(self):
        pasta = filedialog.askdirectory()
        if pasta:
            self.path_imagens.set(pasta)

    def run_setup_rois(self):
        try:
            p = Path(self.path_imagens.get())
            criar_config_rois(self.angulo.get(), self.espelhar.get(), 0.25, "config_rois.json", p)
            messagebox.showinfo("Sucesso", "Configuração de ROIs concluída!")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def run_tracking(self):
        def task():
            try:
                self.status_label.configure(text="Status: Processando Tracking...", text_color="orange")
                p = Path(self.path_imagens.get())
                criar_tracking_geometrico(self.angulo.get(), self.espelhar.get(), 0.25, "tracking_refined.json", p)
                self.status_label.configure(text="Status: Tracking Concluído!", text_color="green")
            except Exception as e:
                self.status_label.configure(text=f"Erro: {str(e)}", text_color="red")
        
        threading.Thread(target=task).start()

    def run_main_process(self):
        def task():
            try:
                self.status_label.configure(text="Status: Renderizando Vídeo...", text_color="orange")
                p = Path(self.path_imagens.get())
                criar_filme_com_audio(
                    JSON_ROIS="config_rois.json",
                    JSON_REFINADO="tracking_refined.json",
                    PASTA_IMAGENS=p,
                    FFMPEG_PATH=self.ffmpeg_path.get(),
                    FPS=int(self.fps.get()),
                    VISUALIZAR_DEBUG=False,
                    LIMITE_TESTE_SEGUNDOS=self.limite_teste.get(),
                    SCALE_FACTOR=self.scale_factor.get()
                )
                self.status_label.configure(text="Status: VÍDEO CONCLUÍDO!", text_color="cyan")
                messagebox.showinfo("Finalizado", "O filme foi gerado com sucesso!")
            except Exception as e:
                self.status_label.configure(text=f"Erro: {str(e)}", text_color="red")

        threading.Thread(target=task).start()
    
    def run_audio_frequency_analysis(self):
        def task():
            try:
                self.status_label.configure(text="Status: Analisando Frequências...", text_color="orange")
                analisar_ruido("audio_clean.wav")
                self.status_label.configure(text="Status: Análise de Frequência Concluída!", text_color="green")
                messagebox.showinfo("Análise Concluída", "A análise de frequência foi concluída! Verifique o arquivo 'analise_frequencia.png'.")
            except Exception as e:
                self.status_label.configure(text=f"Erro: {str(e)}", text_color="red")

        threading.Thread(target=task).start()
    
    def run_transition_detection(self):
        def task():
            try:
                self.status_label.configure(text="Status: Detectando Transições...", text_color="orange")
                detectar_saltos_com_grafico()
                self.status_label.configure(text="Status: Detecção de Transições Concluída!", text_color="green")
                messagebox.showinfo("Detecção Concluída", "A detecção de transições foi concluída! Verifique o gráfico interativo.")
            except Exception as e:
                self.status_label.configure(text=f"Erro: {str(e)}", text_color="red")
        threading.Thread(target=task).start()

    def criar_visualizacao_cascata_mega_overlap(self):
        """Etapa 6: Visualização pedagógica com cascata e forma de onda."""
        scanner = ArqueologiaSonora(self.path_imagens.get())
        scanner.analisar_pipeline_frame(0) # Etapa 1 e 2 integrada
        scanner.compilar_audio(limite_frames=240) # Etapa 3 e 4 integrada


if __name__ == "__main__":
    app = ScannerInterface()
    app.mainloop()