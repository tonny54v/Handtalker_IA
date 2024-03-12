# Importar librerias
import math
import cv2
import mediapipe as mp
import time

# Creacion de una clase
class detectormanos():
    #Parametros de deteccion
    def __init__(self, mode=False, maxManos = 2, model_complexity=1, Confdeteccion = 0.5, Confsegui = 0.5):
        self.mode = mode
        self.maxManos = maxManos
        self.compl = model_complexity
        self.Confdeteccion = Confdeteccion
        self.Confsegui = Confsegui

        #Objetos que detectaran y dibujaran manos
        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(self.mode, self.maxManos, self.compl, self.Confdeteccion, self.Confsegui)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4,8,12,16,20]

    #Funcion para encontrar manos
    def encontrarmanos(self, frame, dibujar = True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self.resultados = self.manos.process(imgcolor)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS) #Dibujar las conexiones
        return frame

    #Funcion para encontrar la posicion
    def encontrarposicion(self, frame, manoNum = 0, dibujarPuntos = True, dibujarBox = True, color = []):
        xLista = []
        yLista = []
        bbox = []
        player = 0
        self.lista = []
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[manoNum]
            prueba = self.resultados.multi_hand_landmarks
            player = len(prueba)

            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                xLista.append(cx)
                yLista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujarPuntos:
                    cv2.circle(frame,(cx, cy), 3, (0,0,0), cv2.FILLED)

            xmin, xmax = min(xLista), max(xLista)
            ymin, ymax = min(yLista), max(yLista)
            bbox = xmin, ymin, xmax, ymax
            if dibujarBox:
                #Dibuja cuadro
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), color, 2)
        return self.lista, bbox, player

    #Funcion para detectar y dibujar los dedos arriba
    def dedosarriba(self):
        dedos = []
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0]-1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        for id in range (1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id]-2][2]:
                dedos.append(1)
            else:
                dedos.append(0)
        return dedos

    #funcion para detectar la distancia entre dedos
    def distancia(self, p1, p2, frame, dibujar = True, r = 15, t = 3):
        x1, y1 = self.lista[p1][1:]
        x2, y2 = self.lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if dibujar:
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), t)
            cv2.circle(frame, (x1, y1), r, (0,0,255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]

#Funcion principal
def main():
    ptiempo = 0
    ctiempo = 0

    #leer camara web
    cap = cv2.VideoCapture(0)
    #Crear el objeto
    detector = detectormanos()

    #Proceso de deteccion de manos
    while True:
        ret, frame = cap.read()
        #Una vez obtenida la imagen se envia
        frame = detector.encontrarposicion(frame)
        #if len(lista) !=0:
            #print(lista[4])

        #Mostrar fps
        ctiempo = time.time()
        fps = 1 / (ctiempo - ptiempo)
        ptiempo = ctiempo

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)

        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()