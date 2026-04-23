import numpy as np
import cv2
import os

def load_colorization_model(model_dir="."):
    """
    Carrega o modelo de rede neural para colorização.
    """
    proto = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
    model = os.path.join(model_dir, "colorization_release_v2.caffemodel")
    points = os.path.join(model_dir, "pts_in_hull.npy")

    # Carrega o modelo de rede neural
    net = cv2.dnn.readNetFromCaffe(proto, model)
    pts = np.load(points)

    # Adiciona os centros de cluster como camadas de convolução 1x1
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full((1, 313), 2.606, dtype="float32")]
    
    return net

def colorize_image(image, net):
    """
    Aplica a colorização em uma imagem (em escala de cinza/P&B).
    Pode receber o caminho da imagem ou o numpy array (cv2 image).
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image}")

    # Carrega e processa a imagem
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Redimensiona para o que a rede espera (224x224) e extrai o canal L
    resized = cv2.resize(lab, (224, 224))
    L = resized[:, :, 0]
    L -= 50 # Ajuste de média comum para esse modelo

    # Faz a predição dos canais 'a' e 'b'
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Redimensiona 'ab' para o tamanho original e combina com o L original
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L_orig = lab[:, :, 0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)

    # Converte de volta para BGR (formato padrão do OpenCV)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255 * np.clip(colorized, 0, 1)).astype("uint8")
    
    return colorized

if __name__ == "__main__":
    # Teste isolado do script
    try:
        net_model = load_colorization_model()
        image_path = "foto_antiga.jpg"
        
        if os.path.exists(image_path):
            img_original = cv2.imread(image_path)
            img_colorida = colorize_image(img_original, net_model)
            
            cv2.imshow("Original", img_original)
            cv2.imshow("Colorida", img_colorida)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Imagem de teste não encontrada: {image_path}")
            
    except Exception as e:
        print(f"Erro ao executar a colorização: {e}")
