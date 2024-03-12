# Libreras
import cv2
import os
from ultralytics import YOLO

# importar la clase seguimiento manos
import SeguimientoManos as sm

# Lectura de la camara
cap = cv2.VideoCapture(0)
# Cambiar la resolucion a HD
cap.set(3, 1280)
cap.set(4, 720)

# Leer nuestro modelo
model = YOLO('C.pt')

# Declarar detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    # Realizar la lectura de la captura
    ret, frame = cap.read()

    # Extraer informacion de la mano
    frame = detector.encontrarmanos(frame, dibujar=False)

    # Posicion de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, manoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0,255,0])

    # Si hay mano
    if mano == 1:
        # Extraer la informacion del cuadro
        xmin, ymin, xmax, ymax = bbox

        # Asignamos margen
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        # Realizar recorte
        recorte = frame[ymin:ymax, xmin:xmax]

        # Redimensionamiento
        #recorte = cv2.resize(recorte, (500,500), interpolation=cv2.INTER_CUBIC)

        # Extraer resultados
        resultados = model.predict(recorte, conf=0.55)

        # Si hay resultados
        if len(resultados) != 0:
            # Iteramos
            for results in resultados:
                masks = results.masks
                coordenadas = masks

                anotaciones = resultados[0].plot()

        cv2.imshow("Recorte", anotaciones)

    # Mostrar fps
    cv2.imshow("Lenguaje vocales", frame)
    # Leer nuestro teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
