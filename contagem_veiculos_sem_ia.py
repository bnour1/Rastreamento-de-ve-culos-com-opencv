import cv2

# Inicialização do vídeo
video_path = 'video4.mp4'  # Define o caminho para o arquivo de vídeo
cap = cv2.VideoCapture(video_path)  # Abre o vídeo especificado

# Inicialização do subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
# Cria um objeto de subtrator de fundo. O algoritmo MOG2 (Mixture of Gaussian Models 2) é usado para identificar pixels em movimento.
# history=500: Define o número de frames para a memória histórica, influenciando na detecção de fundo.
# varThreshold=50: Aumenta a sensibilidade para detectar movimento.
# detectShadows=True: Habilita a detecção de sombras (opcional).

# Parâmetros para a linha de contagem
line_position = 400  # Posição vertical da linha de contagem na imagem
offset = 6  # Tolerância (em pixels) para definir se um veículo cruzou a linha de contagem

# Contagem de veículos
vehicle_count = 0  # Variável para armazenar a contagem de veículos

# Função para detectar veículos
def detect_vehicles(frame):
    # Objetivo: Detectar os veículos no frame atual.
    fgmask = fgbg.apply(frame)  # Aplica o subtrator de fundo para obter uma máscara de movimento.
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)  # Limiariza a máscara de movimento para um resultado binário (preto e branco).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Cria um elemento estruturante para a operação de fechamento.
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)  # Aplica uma operação de fechamento para remover pequenos buracos na máscara de movimento.
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Encontra contornos na máscara de movimento.
    detections = []  # Lista para armazenar as detecções de veículos.
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Verifica se a área do contorno é maior que 500 (filtra pequenos objetos)
            x, y, w, h = cv2.boundingRect(cnt)  # Obtém a caixa delimitadora (x, y, w, h)
            detections.append((x, y, w, h))  # Adiciona à lista detections
    return detections  # Retorna a lista detections contendo as posições das caixas delimitadoras dos veículos.

# Função para e contornar os veículos com os retângulos contar veículos
def draw_and_count(frame, detections):
    # Objetivo: Desenhar retângulos ao redor dos veículos e atualizar a contagem.
    global vehicle_count  # Declara a variável vehicle_count como global
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenha um retângulo verde ao redor de cada veículo.
        if (line_position - offset) < (y + h) < (line_position + offset):  # Verifica se o centro do veículo cruzou a linha de contagem
            vehicle_count += 1  # Incrementa a contagem vehicle_count
            detections.remove((x, y, w, h))  # Remove a detecção da lista
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)  # Desenha a linha de contagem na imagem.
    cv2.putText(frame, f"Veiculos: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Escreve a contagem de veículos na imagem.
    return frame  # Retorna o frame atualizado com os retângulos e a contagem.

while True:
    # Objetivo: Processa cada frame do vídeo.
    ret, frame = cap.read()  # Lê o próximo frame do vídeo.
    if not ret:  # Se ret for False (fim do vídeo), sai do loop.
        break
    
    detections = detect_vehicles(frame)  # Chama a função para detectar veículos.
    frame = draw_and_count(frame, detections)  # Chama a função para desenhar e contar veículos.
    
    cv2.imshow("Contagem de Veículos", frame)  # Exibe o frame na tela.
    
    if cv2.waitKey(30) & 0xFF == ord('q'):  # Espera por um tempo definido (30 ms). Se a tecla 'q' for pressionada, sai do loop.
        break

cap.release()  # Libera os recursos do vídeo.
cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV.