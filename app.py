from ultralytics import YOLO
import cv2

# 1. Carregar o modelo (o ficheiro yolov8n.pt será usado)
modelo = YOLO("yolov8n.onnx")

# 2. Iniciar a câmara web (0 é a câmara principal)
camara = cv2.VideoCapture(0)

# Verificar se a câmara abriu corretamente
if not camara.isOpened():
    print("Erro: Não foi possível aceder à câmara.")
    exit()

print("A iniciar a câmara... Pressione a tecla 'q' para sair.")

# 3. Ciclo infinito para ler continuamente a câmara
while True:
    # Ler o frame (a fotografia atual) da câmara
    sucesso, frame = camara.read()
    
    if not sucesso:
        print("Erro ao ler a imagem da câmara.")
        break
        
    # 4. Fazer a deteção no frame atual
    # stream=True otimiza o uso de memória em tempo real
    resultados = modelo.track(frame, persist=True, classes=[0, 16, 73], device=0)
    
    # 5. Desenhar as marcações
    for resultado in resultados:
        # A função plot() desenha as caixas e os rótulos automaticamente
        frame_marcado = resultado.plot()
        
    # 6. Mostrar o resultado num ecrã
    cv2.imshow("Deteção em Tempo Real", frame_marcado)
    
    # 7. Aguardar 1 milissegundo e verificar se a tecla 'q' foi pressionada para fechar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Limpeza: Desligar a câmara e fechar a janela
camara.release()
cv2.destroyAllWindows()