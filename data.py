# Libreras
import cv2
import os

# importar la clase seguimiento manos
import SeguimientoManos as sm

# creacion de la carpeta
nombre = 'Letra_X'
direccion = 'D:/proyecto_handtalker_IA/Handtalker_ia/Entrenamiento_IA/Data'
carpeta = direccion + '/' + nombre

# Si no esta creada la carpeta se crea
if not os.path.exists(carpeta):
    print("Carpeta creada: ", carpeta)
    # Crea la carpeta
    os.makedirs(carpeta)

# Lectura de la camara
cap = cv2.VideoCapture(0)
# Cambiar la resolucion a HD
cap.set(3, 1280)
cap.set(4, 720)

# Declarar contador
cont = 0

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

        # Almacenar nuestras imagenes
        cv2.imwrite(carpeta + "/X_{}.jpg".format(cont), recorte)

        # Aumentamos contador
        cont = cont + 1

        cv2.imshow("Recorte", recorte)

        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0,255,0], 2)


    # Mostrar fps
    cv2.imshow("Lenguaje vocales", frame)
    # Leer nuestro teclado
    t = cv2.waitKey(1)
    if t == 27 or cont == 100:
        break

cap.release()
cv2.destroyAllWindows()