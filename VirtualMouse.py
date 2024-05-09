import cv2
import numpy as np
import HandDetectionModule
import time
import autopy

wCam, hCam = 1280, 720
frameRh = 200
frameRw = 250
smoothening = 7

x1, x2, y1, y2 = 0, 0, 0, 0

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetectionModule.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
toggleValue = True

stopFlag = True
while stopFlag:

    # Gasire pozitii maini
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Pozitia pentru varful degetelor aratator si mijlociu ( click )
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # Vector in care vom avea ce degete sunt ridicate
    fingers = detector.fingersUp()
    cv2.rectangle(img, (frameRw, frameRh), (wCam - frameRw, hCam - frameRh),
                  (255, 0, 255), 2)
    print(fingers)
    # Daca avem doar deget aratator vom misca cursorul
    if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:

        # Raportam coordonatele degetelor din camera la cele de pe ecran
        x3 = np.interp(x1, (frameRw, wCam - frameRw), (0, wScr))
        y3 = np.interp(y1, (frameRh, hCam - frameRh), (0, hScr))

        # Scapam de jitter (zgomotul creat de faptul ca mana nu poate sta perfect stationara,
        # de imperfectiunile camerei) folosind un coeficient de smoothening
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # Pozitia degetelor
        # print(clocX, clocY)

        # miscam mouse-ul propriu zis ( avem efect in oglinda pe verticala, deci scadem
        # din width x-ul nostru )
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # Daca avem degetul aratator si mijlociu ridicate vom da click
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 0:

        # Cautam distanta dintre cele doua degete (vrem sa dam click doar daca distanta asta
        # va fi mai mica decat o anumita valoare
        length, img, lineInfo = detector.findDistance(8, 12, img)
        print(length)

        # Daca e mai mica decat o valoare aleasa, dam click
        if length < 25:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                       15, (0, 255, 0), cv2.FILLED)
        autopy.mouse.click()

    # Daca doar degetul mare este ridicat trecem in modul de click apasat
    if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1 and fingers[0] == 0:
        # toggleValue = not toggleValue
        if toggleValue:
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
            toggleValue = False

        x3 = np.interp(x1, (frameRw, wCam - frameRw), (0, wScr))
        y3 = np.interp(y1, (frameRh, hCam - frameRh), (0, hScr))

        # Scapam de jitter (zgomotul creat de faptul ca mana nu poate sta perfect stationara,
        # de imperfectiunile camerei) folosind un coeficient de smoothening
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # Pozitia degetelor
        # print(clocX, clocY)

        # miscam mouse-ul propriu zis ( avem efect in oglinda pe verticala, deci scadem
        # din width x-ul nostru )
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
    else:
        if not toggleValue:
            toggleValue = True
            autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)

    # Frame rate-ul camerei
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # Imaginea
    cv2.imshow("Hand Recognition - Proiect AM (TMEAI) Neleptcu Daniel-Andrei", img)
    cv2.waitKey(1)
