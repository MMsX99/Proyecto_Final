import mediapipe as mp
import cv2
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

gesto_actual="None"
c_errores=0


with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    imagen1 = cv2.imread("C:/Users/ASUS/Documents/universidad distrital/Proyectos_aplicada/Proyecto_Hands_basura/imagenes/gesto_sostener3.png")
    assert not isinstance(imagen1,type(None)), 'image not found'
    #print(imagen1)
    height, width, _ = imagen1.shape
    imagen1 = cv2.flip(imagen1, 1)
    #-----------------

    imagen1_rgb = cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB)

    resultados= hands.process(imagen1_rgb)




    
    if resultados.multi_hand_landmarks is not None:
        ##############################

        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                imagen1, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,0), thickness=4, circle_radius=5),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=4)
            )

            #acceder a los puntos
            #x4=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x )
            #y4=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y)
            #print(x4)
            #print(y4)

            puntos=np.array([[punto.x, punto.y, punto.z] for punto in hand_landmarks.landmark ]).T
            #print(puntos)

            #VECTORES MCP_0
            vMCP_0=np.array([[puntos[0,x]-puntos[0,0], puntos[1,x]-puntos[1,0], puntos[2,x]-puntos[2,0] ] for x in (5,9,13,17) ]).T
            #print(vMCP_0)

            
            #x1PIP_MCP=np.array([puntos[0,x] for x in  (6,10,14,18) ])
            #x0PIP_MCP=np.array([puntos[0,x] for x in (5,9,13,17) ])
            #y1PIP_MCP=np.array([puntos[1,y] for y in  (6,10,14,18) ])
            #y0PIP_MCP=np.array([puntos[1,y] for y in (5,9,13,17) ])
            #xPIP_MCP=x1PIP_MCP-x0PIP_MCP
            #yPIP_MCP=y1PIP_MCP-y0PIP_MCP
            
            #vPIP_MCP=np.array([xPIP_MCP,yPIP_MCP])
            #print(vPIP_MCP)
            
            #vectores de todas las falanges:

            pPIP=np.array([[puntos[0,x], puntos[1,x], puntos[2,x]] for x in (6,10,14,18) ]).T
            pMCP=np.array([[puntos[0,x], puntos[1,x], puntos[2,x]] for x in (5,9,13,17) ]).T
            vPIP_MCP=pPIP-pMCP
            #print(vPIP_MCP)
            #print('\n')
            
            pDIP=np.array([[puntos[0,x], puntos[1,x], puntos[2,x]] for x in (7,11,15,19) ]).T
            vDIP_PIP= pDIP-pPIP
            #print(vDIP_PIP, end='\n')
            #print('\n')

            pTIP=np.array([[puntos[0,x], puntos[1,x], puntos[2,x]] for x in (8,12,16,20) ]).T
            vTIP_DIP= pTIP-pDIP
            #print(vTIP_DIP)
            #print('\n')


            #para cada dedo, coseno entre vectores:

            vMCP_index=[vMCP_0[0,0], vMCP_0[1,0], vMCP_0[2,0]]
            vMCP_middle=[vMCP_0[0,1], vMCP_0[1,1], vMCP_0[2,1]]
            vMCP_ring=[vMCP_0[0,2], vMCP_0[1,2], vMCP_0[2,2]]
            vMCP_pinky=[vMCP_0[0,3], vMCP_0[1,3], vMCP_0[2,3]]

            vPIP_index=[vPIP_MCP[0,0], vPIP_MCP[1,0], vPIP_MCP[2,0]]
            vPIP_middle=[vPIP_MCP[0,1], vPIP_MCP[1,1], vPIP_MCP[2,1]]
            vPIP_ring=[vPIP_MCP[0,2], vPIP_MCP[1,2], vPIP_MCP[2,2]]
            vPIP_pinky=[vPIP_MCP[0,3], vPIP_MCP[1,3], vPIP_MCP[2,3]]

            vDIP_index=[vDIP_PIP[0,0], vDIP_PIP[1,0], vDIP_PIP[2,0]]
            vDIP_middle=[vDIP_PIP[0,1], vDIP_PIP[1,1], vDIP_PIP[2,1]]
            vDIP_ring=[vDIP_PIP[0,2], vDIP_PIP[1,2], vDIP_PIP[2,2]]
            vDIP_pinky=[vDIP_PIP[0,3], vDIP_PIP[1,3], vDIP_PIP[2,3]]

            vTIP_index=[vTIP_DIP[0,0], vTIP_DIP[1,0], vDIP_PIP[2,0]]
            vTIP_middle=[vTIP_DIP[0,1], vTIP_DIP[1,1], vDIP_PIP[2,1]]
            vTIP_ring=[vTIP_DIP[0,2], vTIP_DIP[1,2], vDIP_PIP[2,2]]
            vTIP_pinky=[vTIP_DIP[0,3], vTIP_DIP[1,3], vDIP_PIP[2,3]]

            #

            cosPIP_MCP_index=np.dot(vPIP_index, vMCP_index) / (np.linalg.norm(vPIP_index)*np.linalg.norm(vMCP_index))
            cosPIP_MCP_middle=np.dot(vPIP_middle, vMCP_middle) / (np.linalg.norm(vPIP_middle)*np.linalg.norm(vMCP_middle))
            cosPIP_MCP_ring=np.dot(vPIP_ring, vMCP_ring) / (np.linalg.norm(vPIP_ring)*np.linalg.norm(vMCP_ring))
            cosPIP_MCP_pinky=np.dot(vPIP_pinky, vMCP_pinky) / (np.linalg.norm(vPIP_pinky)*np.linalg.norm(vMCP_pinky))

            cosDIP_PIP_index=np.dot(vDIP_index, vPIP_index) / (np.linalg.norm(vDIP_index)*np.linalg.norm(vPIP_index))
            cosDIP_PIP_middle=np.dot(vDIP_middle, vPIP_middle) / (np.linalg.norm(vDIP_middle)*np.linalg.norm(vPIP_middle))
            cosDIP_PIP_ring=np.dot(vDIP_ring, vPIP_ring) / (np.linalg.norm(vDIP_ring)*np.linalg.norm(vPIP_ring))
            cosDIP_PIP_pinky=np.dot(vDIP_pinky, vPIP_pinky) / (np.linalg.norm(vDIP_pinky)*np.linalg.norm(vPIP_pinky))

            cosTIP_DIP_index=np.dot(vTIP_index, vDIP_index) / (np.linalg.norm(vTIP_index)*np.linalg.norm(vDIP_index))
            cosTIP_DIP_middle=np.dot(vTIP_middle, vDIP_middle) / (np.linalg.norm(vTIP_middle)*np.linalg.norm(vDIP_middle))
            cosTIP_DIP_ring=np.dot(vTIP_ring, vDIP_ring) / (np.linalg.norm(vTIP_ring)*np.linalg.norm(vDIP_ring))
            cosTIP_DIP_pinky=np.dot(vTIP_pinky, vDIP_pinky) / (np.linalg.norm(vTIP_pinky)*np.linalg.norm(vDIP_pinky))

            #Vector normal a la palma  de la mano:
            vPlano1=(vMCP_0.T)[0]
            #print(vPlano1)
            vPlano2=(vMCP_0.T)[2]
            #print(vPlano2)
            vNormal=np.cross(vPlano1, vPlano2)
            vNormal_mag=np.linalg.norm(vNormal)
            vNormal=-vNormal*(1/vNormal_mag)/20
            vNormal_mag=np.linalg.norm(vNormal)
            #print(vNormal_mag)

            vCentro=((vMCP_0.T)[1])/2
            pCentro=vCentro + ((puntos.T)[0])
            pCentro_xy=[int(pCentro[0]*width), int(pCentro[1]*height)]
            cv2.circle(imagen1, list(pCentro_xy),  8, (0,0,255), 4)

            pNormal=vNormal+pCentro
            pNormal_xy=[int(pNormal[0]*width), int(pNormal[1]*height)]
            cv2.arrowedLine(imagen1, list(pCentro_xy), list(pNormal_xy),(255-(4*60),(4*180)%256,4*60), 6)

            #np.linalg.norm(x)*np.linalg.norm(y)
            print('coseno del angulo entre falanges [index, middle, ring, pinky]:')
            print(' PIP_MCP_0: ')
            print(cosPIP_MCP_index,
                cosPIP_MCP_middle,
                cosPIP_MCP_ring,
                cosPIP_MCP_pinky)

            print(' DIP_PIP_MCP: ')
            print(cosDIP_PIP_index,
                cosDIP_PIP_middle,
                cosDIP_PIP_ring,
                cosDIP_PIP_pinky)
            
            print(' TIP_DIP_TIP: ')
            print(cosTIP_DIP_index,
                cosTIP_DIP_middle,
                cosTIP_DIP_ring,
                cosTIP_DIP_pinky)


            #DIFERENCIAR GESTOS PASO 1, SOSTENER O SOLTAR:
            #tolerancias 
            
            ##SOSTENER:
   
            min_cosPIP_MCP_index=-0.5
            min_cosPIP_MCP_middle=-0.5
            min_cosPIP_MCP_ring=-0.5
            min_cosPIP_MCP_pinky=-0.5
            
            min_cosDIP_PIP_index=-0.5
            min_cosDIP_PIP_middle=-0.5
            min_cosDIP_PIP_ring=-0.5
            min_cosDIP_PIP_pinky=-0.5
            
            min_cosTIP_DIP_index=-0.5
            min_cosTIP_DIP_middle=-0.5
            min_cosTIP_DIP_ring=-0.5
            min_cosTIP_DIP_pinky=-0.5

            max_cosPIP_MCP_index=0.95
            max_cosPIP_MCP_middle=0.95
            max_cosPIP_MCP_ring=0.95
            max_cosPIP_MCP_pinky=0.95

            max_cosDIP_PIP_index=0.95
            max_cosDIP_PIP_middle=0.95
            max_cosDIP_PIP_ring=0.95
            max_cosDIP_PIP_pinky=0.95

            max_cosTIP_DIP_index=0.95
            max_cosTIP_DIP_middle=0.96
            max_cosTIP_DIP_ring=0.95
            max_cosTIP_DIP_pinky=0.95

            
            c_errores=0
            while True:
            
                if not (cosPIP_MCP_index >= min_cosPIP_MCP_index and cosPIP_MCP_index <= max_cosPIP_MCP_index) :
                    c_errores = c_errores +1
                    if c_errores == 2: break 
                    
                if not (cosPIP_MCP_middle >= min_cosPIP_MCP_middle and cosPIP_MCP_middle<=max_cosPIP_MCP_middle):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosPIP_MCP_ring >= min_cosPIP_MCP_ring and cosPIP_MCP_ring <= max_cosPIP_MCP_ring):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosPIP_MCP_pinky >= min_cosPIP_MCP_pinky and cosPIP_MCP_pinky <= max_cosPIP_MCP_pinky):
                    c_errores = c_errores +1
                    if c_errores == 2: break            

                if not (cosDIP_PIP_index >= min_cosDIP_PIP_index and cosDIP_PIP_index <= max_cosDIP_PIP_index):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosDIP_PIP_middle >= min_cosDIP_PIP_middle and cosDIP_PIP_middle <= max_cosDIP_PIP_middle):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosDIP_PIP_ring >= min_cosDIP_PIP_ring and cosDIP_PIP_ring <= max_cosDIP_PIP_ring):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosDIP_PIP_pinky >= min_cosDIP_PIP_pinky and cosDIP_PIP_pinky <= max_cosDIP_PIP_pinky):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                
                if not (cosTIP_DIP_index >= min_cosTIP_DIP_index and cosTIP_DIP_index <= max_cosTIP_DIP_index):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosTIP_DIP_middle >= min_cosTIP_DIP_middle and cosTIP_DIP_middle <= max_cosTIP_DIP_middle):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosTIP_DIP_ring >= min_cosTIP_DIP_ring and cosTIP_DIP_ring <= max_cosTIP_DIP_ring):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosTIP_DIP_pinky >= min_cosTIP_DIP_pinky and cosTIP_DIP_pinky <= max_cosTIP_DIP_pinky):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                
                gesto_actual="Sostener"
                break


            #########
            #########
            #SOLTAR:
            min_cosPIP_MCP_index=0.85
            min_cosPIP_MCP_middle=0.9
            min_cosPIP_MCP_ring=0.9
            min_cosPIP_MCP_pinky=0.55
            
            min_cosDIP_TIP_index=0.9
            min_cosDIP_TIP_middle=0.9
            min_cosDIP_TIP_ring=0.9
            min_cosDIP_TIP_pinky=0.9
            
            min_cosTIP_DIP_index=0.92
            min_cosTIP_DIP_middle=0.92
            min_cosTIP_DIP_ring=0.9
            min_cosTIP_DIP_pinky=0.92

            max_cosPIP_MCP_index=1
            max_cosPIP_MCP_middle=1
            max_cosPIP_MCP_ring=1
            max_cosPIP_MCP_pinky=1

            max_cosDIP_PIP_index=1
            max_cosDIP_PIP_middle=1
            max_cosDIP_PIP_ring=1
            max_cosDIP_PIP_pinky=1

            max_cosTIP_DIP_index=1
            max_cosTIP_DIP_middle=1
            max_cosTIP_DIP_ring=1
            max_cosTIP_DIP_pinky=1

            c_errores=0   
            while True:
            
                if not (cosPIP_MCP_index >= min_cosPIP_MCP_index and cosPIP_MCP_index <= max_cosPIP_MCP_index) :
                    c_errores = c_errores +1
                    if c_errores == 2: break 
                    
                if not (cosPIP_MCP_middle >= min_cosPIP_MCP_middle and cosPIP_MCP_middle<=max_cosPIP_MCP_middle):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosPIP_MCP_ring >= min_cosPIP_MCP_ring and cosPIP_MCP_ring <= max_cosPIP_MCP_ring):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosPIP_MCP_pinky >= min_cosPIP_MCP_pinky and cosPIP_MCP_pinky <= max_cosPIP_MCP_pinky):
                    c_errores = c_errores +1
                    if c_errores == 2: break            

                if not (cosDIP_PIP_index >= min_cosDIP_PIP_index and cosDIP_PIP_index <= max_cosDIP_PIP_index):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosDIP_PIP_middle >= min_cosDIP_PIP_middle and cosDIP_PIP_middle <= max_cosDIP_PIP_middle):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosDIP_PIP_ring >= min_cosDIP_PIP_ring and cosDIP_PIP_ring <= max_cosDIP_PIP_ring):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosDIP_PIP_pinky >= min_cosDIP_PIP_pinky and cosDIP_PIP_pinky <= max_cosDIP_PIP_pinky):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                
                if not (cosTIP_DIP_index >= min_cosTIP_DIP_index and cosTIP_DIP_index <= max_cosTIP_DIP_index):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosTIP_DIP_middle >= min_cosTIP_DIP_middle and cosTIP_DIP_middle <= max_cosTIP_DIP_middle):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosTIP_DIP_ring >= min_cosTIP_DIP_ring and cosTIP_DIP_ring <= max_cosTIP_DIP_ring):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                if not (cosTIP_DIP_pinky >= min_cosTIP_DIP_pinky and cosTIP_DIP_pinky <= max_cosTIP_DIP_pinky):
                    c_errores = c_errores +1
                    if c_errores == 2: break
                
                gesto_actual="Soltar"
                break

            print('\n')            
            print("GESTO DE LA IMAGEN:", gesto_actual)    
            print('\n')


            ##########################


    #----------------
    imagen1= cv2.flip(imagen1, 1)
resized =   cv2.resize(imagen1, (1080,720), interpolation = cv2.INTER_AREA) 
cv2.imshow("Imagen1", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()