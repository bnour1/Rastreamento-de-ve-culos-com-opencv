
import cv2
import numpy as np

# Inicialização do vídeo
video_path = 'Test Video.mp4'
cap = cv2.VideoCapture(video_path)

# Inicialização do subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Parâmetros para a linha de contagem
line_position = 400
offset = 6  # tolerância para a contagem

# Contagem de veículos
vehicle_count = 0

# Função para detectar veículos
def detect_vehicles(frame):
    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))
    return detections

# Função para desenhar e contar veículos
def draw_and_count(frame, detections):
    global vehicle_count
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if (line_position - offset) < (y + h) < (line_position + offset):
            vehicle_count += 1
            detections.remove((x, y, w, h))
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)
    cv2.putText(frame, f"Veículos: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = detect_vehicles(frame)
    frame = draw_and_count(frame, detections)
    
    cv2.imshow("Contagem de Veículos", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
