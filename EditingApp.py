import math
import os
import random
import skimage
from PIL import ImageColor
import cv2
import numpy as np
from tkinter import filedialog, font
from tkinter import *
import tkinter.messagebox as msg
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from collections import deque
import copy
from tkinter.colorchooser import askcolor
import threading
import subprocess
import importlib

global pixelMatrix, height, width, lolz, myimage2, previewHeight, previewWidth, pixelMatrixStack
global undoValue, resetMatrix, pixelMatrixSlide, pixelMatrixSlideOriginal, actualCvtCOLOR, color
global choose_size_button, old_x, old_y, paintBrush, line_width, lineWidthFlag, tipBrush, options
global densityScale, angleScale, density, angle_range, CropXInit, CropYInit, x1, x2, y1, y2
global pixelMatrix1, pixelMatrix2, pixelMatrix3, pixelMatrix4, rec
global sliderWin, labelSharpness, labelSolarization, scaleSharpness, scaleSolarization
global labelBrightness, labelContrast, labelHue, labelSaturation, labelBlurMod, labelBlurSigma
global scaleBrightness, scaleContrast, scaleHue, scaleSaturation, scaleBlurMod, scaleBlurSigma
global lastFunctionCall, sliderWinOpen, wasMoved, validResize, qualityScale, qualityLabel
global resizeCoeff, resizeScale, resizeWidth, resizeHeight, varList, addKernelWin
script_process = None


def start_script():
    return subprocess.Popen(["python", "VirtualMouse.py"])


def start_thread():
    global script_process
    script_process = start_script()


def ResizeWithAspectRatio(image, widths=None, heights=None, inter=cv2.INTER_AREA):
    global lastFunctionCall
    lastFunctionCall = 'ResizeWithAspectRatio'
    (h, w) = image.shape[:2]

    if widths is None and heights is None:
        return image
    if widths is None:
        r = heights / float(h)
        dim = (int(w * r), heights)
    else:
        r = widths / float(w)
        dim = (widths, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def controller(brightnessS=255, contrasts=127):
    global pixelMatrixSlideOriginal
    brightnessS = int((brightnessS - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrasts = int((contrasts - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightnessS != 0:
        if brightnessS > 0:
            shadow = brightnessS
            maxq = 255
        else:
            shadow = 0
            maxq = 255 + brightnessS

        al_pha = (maxq - shadow) / 255
        ga_mma = shadow

        cal = cv2.addWeighted(pixelMatrixSlideOriginal, al_pha,
                              pixelMatrixSlideOriginal, 0, ga_mma)
    else:
        cal = pixelMatrixSlideOriginal

    if contrasts != 0:
        Alpha = float(131 * (contrasts + 127)) / (127 * (131 - contrasts))
        Gamma = 127 * (1 - Alpha)

        cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)

    return cal


def blurImage(blur, SigmaX):
    global pixelMatrix1
    cal = cv2.GaussianBlur(pixelMatrix1, (2 * blur + 1, 2 * blur + 1), SigmaX)
    return cal


def sharpenImage(sharpness):
    global pixelMatrix2
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]]) * (sharpness / 30)
    kernel[1][1] += (1 - sharpness / 30)
    cal = cv2.filter2D(pixelMatrix2, -1, kernel)
    return cal


def adjust_hue_saturation(hue, saturation):
    global actualCvtCOLOR, pixelMatrix3

    hsv = cv2.cvtColor(pixelMatrix3, cv2.COLOR_RGB2HSV)

    hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return adjusted_image


def solarization(threshold):
    global pixelMatrix4
    r, g, b = cv2.split(pixelMatrix4)
    r = np.where(r > threshold, 255 - r, r)
    g = np.where(g > threshold, 255 - g, g)
    b = np.where(b > threshold, 255 - b, b)
    sol = cv2.merge((r, g, b))
    return sol


def sliderChoice(_):
    global previewWidth, pixelMatrixSlide, pixelMatrix
    global pixelMatrix1, pixelMatrix2, pixelMatrix3, pixelMatrix4
    global labelSharpness, labelSolarization
    global labelBrightness, labelContrast, labelHue, labelSaturation, labelBlurMod, labelBlurSigma

    brightnessS = scaleBrightness.get()
    contrasts = scaleContrast.get()
    SigmaX = scaleBlurSigma.get()
    blur = scaleBlurMod.get()
    sharpness = scaleSharpness.get()
    saturation = scaleSaturation.get()
    hue = scaleHue.get()
    solar = scaleSolarization.get()

    labelBrightness["text"] = "Brightness:" + str(brightnessS)
    labelContrast["text"] = "Contrast:" + str(contrasts)
    labelBlurSigma["text"] = "Blur Sigma:" + str(SigmaX)
    labelBlurMod["text"] = "Blur Mod:" + str(blur)

    labelSharpness["text"] = "Sharpness:" + str(sharpness)
    labelSaturation["text"] = "Saturation:" + str(saturation)
    labelHue["text"] = "Hue:" + str(hue)
    labelSolarization["text"] = "Solarization:" + str(solar)

    saturation = saturation / 100.0

    pixelMatrix1 = controller(brightnessS, contrasts)

    pixelMatrix2 = blurImage(blur, SigmaX)

    pixelMatrix3 = sharpenImage(sharpness)

    pixelMatrix4 = adjust_hue_saturation(hue, saturation)

    pixelMatrix = solarization(solar)

    ExportImgPreview()


def vignette():
    global pixelMatrix, pixelMatrixStack
    global lastFunctionCall
    lastFunctionCall = 'vignette'
    rows, cols = pixelMatrix.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols // 3)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows // 3)
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    for i in range(3):
        pixelMatrix[:, :, i] = pixelMatrix[:, :, i] * mask

    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def edgify():
    global lastFunctionCall
    lastFunctionCall = 'edgify'
    global pixelMatrix, pixelMatrixStack, actualCvtCOLOR
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_RGB2GRAY)
    pixelMatrix = cv2.Canny(image=pixelMatrix, threshold1=100, threshold2=200)
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_GRAY2RGB)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def grayscale():
    global lastFunctionCall
    lastFunctionCall = 'grayscale'
    global pixelMatrix, pixelMatrixStack, actualCvtCOLOR
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_RGB2GRAY)
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_GRAY2RGB)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def vertical_flip():
    global lastFunctionCall
    lastFunctionCall = 'vertical_flip'
    global pixelMatrix, pixelMatrixStack
    pixelMatrix = cv2.flip(pixelMatrix, 0)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def horizontal_flip():
    global lastFunctionCall
    lastFunctionCall = 'horizontal_flip'
    global pixelMatrix, pixelMatrixStack
    pixelMatrix = cv2.flip(pixelMatrix, 1)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def sobel(val):
    global lastFunctionCall
    lastFunctionCall = 'sobel'
    global pixelMatrix, pixelMatrixStack
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_RGB2GRAY)

    if val == 'x':
        sobel_x = cv2.Sobel(pixelMatrix, cv2.CV_64F, 1, 0, ksize=3)
        pixelMatrix = cv2.convertScaleAbs(sobel_x)

    if val == 'y':
        sobel_y = cv2.Sobel(pixelMatrix, cv2.CV_64F, 0, 1, ksize=3)
        pixelMatrix = cv2.convertScaleAbs(sobel_y)

    if val == 'f':
        sobel_x = cv2.Sobel(pixelMatrix, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.Sobel(pixelMatrix, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        pixelMatrix = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_GRAY2RGB)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def emboss():
    global pixelMatrix, pixelMatrixStack
    global lastFunctionCall
    lastFunctionCall = 'emboss'
    emboss_kernel = np.array([[-3, -2, 0],
                              [-2, 1, 2],
                              [0, 2, 3]])

    pixelMatrix = cv2.filter2D(pixelMatrix, -1, emboss_kernel)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def sepia():
    global pixelMatrix, pixelMatrixStack
    global lastFunctionCall
    lastFunctionCall = 'sepia'
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    sepia_image = cv2.transform(pixelMatrix, sepia_matrix)

    pixelMatrix = np.clip(sepia_image, 0, 255).astype(np.uint8)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def initCanvas():
    global lineWidthFlag, paintBrush, pixelMatrix, undoValue, line_width, pixelMatrixStack, actualCvtCOLOR
    global tipBrush, height, width
    global lastFunctionCall
    lastFunctionCall = 'initCanvas'
    height = 0
    width = 0
    pixelMatrix = None
    lineWidthFlag = 0
    tipBrush = StringVar()
    paintBrush = IntVar()
    paintBrush.set(0)
    pixelMatrixStack = deque()
    undoValue = 0
    line_width = None


def imgPreview():
    global pixelMatrix, pixelMatrixStack, resetMatrix, actualCvtCOLOR, wasMoved
    global height, width, lastFunctionCall, pixelMatrixSlideOriginal, sliderWinOpen, canvas1
    wasMoved = 0
    sliderWinOpen = 0
    initCanvas()
    root.filename = filedialog.askopenfilename(initialdir="C:/Users/zuch3e/Desktop/Editare de imagini cu hand recognition",
                                               title="Open Image",
                                               filetypes=(
                                                   ("jpg files", "*.jpg"), ("png files", "*.png"),
                                                   ("all files", "*.*")))
    pixelMatrix = cv2.imread(root.filename)
    if pixelMatrix is not None:
        pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_BGR2RGB)
        resetMatrix = copy.deepcopy(pixelMatrix)
        pixelMatrixStack.append(pixelMatrix)
        # print("LEN STACK impreview", len(pixelMatrixStack))
        height = len(pixelMatrix)
        width = len(pixelMatrix[0])
        lastFunctionCall = 'imgPreview'
        ExportImgPreview()
        canvas1.bind('<Button-1>', set_start)
        canvas1.bind('<ButtonRelease-1>', end_line)
        pixelMatrixSlideOriginal = pixelMatrix


def ExportImgPreview():
    global pixelMatrix, myimage2, previewWidth, previewHeight, pixelMatrixStack, undoValue
    global height, width, lastFunctionCall, sliderWinOpen, validImage, sliderWinOpen, sliderWin
    global pixelMatrixSlideOriginal
    if lastFunctionCall != 'sliders':
        pixelMatrixSlideOriginal = copy.deepcopy(pixelMatrix)
    if sliderWinOpen == 0 and wasMoved == 0:
        if lastFunctionCall == 'undo':
            undoValue = undoValue - 1
        else:
            undoValue = undoValue + 1
    # print("Undo Value:", undoValue)
    # if sliderWinOpen == 1 and sliderWin:
    #    sliderChoice(0)
    impil = Image.fromarray(pixelMatrix)
    rows, cols = pixelMatrix.shape[:2]
    height = rows
    width = cols
    previewHeight = 786
    previewWidth = int(786 / height * width)
    if rows < cols:
        previewHeight = int(1400 / width * height)
        previewWidth = 1400

    if height <= 786 and width <= 1400:
        previewHeight = height
        previewWidth = width

    if height > 786 and width > 1400:
        if height / 786 > width / 1400:
            previewHeight = 786
            previewWidth = int(width / height * 786)
        else:
            previewHeight = int(height / width * 1400)
            previewWidth = 1400

    maxi = max(previewWidth, previewHeight)

    impil.thumbnail((maxi, maxi))
    validImage = 1
    myimage2 = ImageTk.PhotoImage(impil)
    canvas1.create_image(702, 395, anchor=CENTER, image=myimage2)


def removeBG():
    global lastFunctionCall
    lastFunctionCall = 'removeBG'
    global pixelMatrix, actualCvtCOLOR
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_RGB2BGR)
    cv2.imwrite("temp.jpg", pixelMatrix, [cv2.IMWRITE_JPEG_QUALITY, 100])
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_BGR2RGB)
    os.remove('temp-removebg.png')
    os.system('removebg --api-key mcAV6Z9HLq4Awu5vmkLVNwz1 temp.jpg')
    os.remove('temp.jpg')
    pixelMatrix = cv2.imread('temp-removebg.png')
    # os.remove('temp-removebg.png')
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_BGR2RGB)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def SaveFinalImage():
    global pixelMatrix, qualityScale
    root.filename2 = filedialog.asksaveasfile(title="Save",
                                              filetypes=(('jpeg', '*.jpg'), ('png', '*.png'), ('all files', '*.*')),
                                              defaultextension='.jpg')

    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_RGB2BGR)
    cv2.imwrite(root.filename2.name, pixelMatrix, [cv2.IMWRITE_JPEG_QUALITY, int(qualityScale.get())])
    pixelMatrix = cv2.cvtColor(pixelMatrix, cv2.COLOR_BGR2RGB)


def updateQualityLabel(_):
    global qualityScale, qualityLabel
    qualityLabel["text"] = "The quality of the saved image will be: " + str(qualityScale.get()) + " %"


def saveimg():
    global validImage, qualityScale, qualityLabel
    if validImage == 1:
        chooseQuality = tk.Toplevel(root)
        chooseQuality.title("Choose Quality")
        chooseQuality.geometry("400x150")
        qualityLabel = tk.Label(chooseQuality, text="The quality of the saved image will be: 95%", justify=CENTER)
        qualityLabel.grid(row=0, column=0)
        qualityScale = tk.Scale(chooseQuality, from_=20, to=100, length=390, width=30, orient=tk.HORIZONTAL,
                                command=updateQualityLabel, showvalue=False)
        qualityScale.set(95)
        qualityScale.grid(row=1, column=0)
        tk.Button(chooseQuality, text="Proceed", command=SaveFinalImage).grid(row=2, column=0, pady=15)
    else:
        tk.messagebox.showerror("Error", "You have not opened an image yet.")


def reset():
    global pixelMatrix, resetMatrix, undoValue, pixelMatrixStack
    global lastFunctionCall, validImage
    if validImage == 1:
        initCanvas()
        pixelMatrix = copy.deepcopy(resetMatrix)

        pixelMatrixStack.append(pixelMatrix)
        ExportImgPreview()
        lastFunctionCall = 'reset'
    else:
        tk.messagebox.showerror("Error", "You have not opened an image yet.")


def undo():
    global pixelMatrix, pixelMatrixStack, lastFunctionCall, sliderWinOpen
    if (lastFunctionCall != 'undo' or undoValue == 1) and len(pixelMatrixStack) != 0 and sliderWinOpen == 0:
        pixelMatrix = copy.deepcopy(pixelMatrixStack.pop())

    if len(pixelMatrixStack) != 0:
        pixelMatrix = copy.deepcopy(pixelMatrixStack.pop())

    # print("LEN STACK:", len(pixelMatrixStack))

    if len(pixelMatrixStack) == 0:
        reset()
        # print("LAST FUNCTION CALL:", lastFunctionCall)
    else:
        # print("LAST FUNCTION CALL:", lastFunctionCall)
        lastFunctionCall = 'undo'
        ExportImgPreview()


def deleteSlider():
    global sliderWin, sliderWinOpen, pixelMatrix, pixelMatrixStack
    print("I am here!")
    sliderWin.destroy()
    sliderWinOpen = 0
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def sliders():
    global pixelMatrix, previewWidth, pixelMatrixSlide, pixelMatrixSlideOriginal
    global sliderWin, labelSharpness, labelSolarization, scaleSharpness, scaleSolarization
    global labelBrightness, labelContrast, labelHue, labelSaturation, labelBlurMod, labelBlurSigma
    global scaleBrightness, scaleContrast, scaleHue, scaleSaturation, scaleBlurMod, scaleBlurSigma
    global lastFunctionCall, sliderWinOpen, validImage
    if sliderWinOpen == 0:
        if validImage == 1:
            sliderWinOpen = 1
            lastFunctionCall = 'sliders'

            pixelMatrixSlide = copy.deepcopy(pixelMatrix)
            pixelMatrixSlideOriginal = copy.deepcopy(pixelMatrixSlide)

            sliderWin = tk.Toplevel(root)
            sliderWin.title("Efecte Dinamice")
            sliderWin.geometry("634x266")
            sliderWin.iconbitmap("strudel_wla_icon.ico")
            sliderWin.resizable(False, False)
            sliderWin.configure(background="#F0F0F0")
            sliderWin.protocol("WM_DELETE_WINDOW", lambda: deleteSlider())

            labelBrightness = tk.Label(sliderWin, text="Brightness:", width=15)
            labelBrightness.grid(row=0, column=0)
            scaleBrightness = tk.Scale(sliderWin, from_=0, to=512, orient=tk.HORIZONTAL, command=sliderChoice,
                                       length=500,
                                       width=25,
                                       background='white', showvalue=False)
            scaleBrightness.set(255)
            scaleBrightness.grid(row=0, column=1)

            labelContrast = tk.Label(sliderWin, text="Contrast:", width=15)
            labelContrast.grid(row=1, column=0)
            scaleContrast = tk.Scale(sliderWin, from_=0, to=255, orient=tk.HORIZONTAL, command=sliderChoice, length=500,
                                     width=25,
                                     background='white', showvalue=False)
            scaleContrast.set(127)
            scaleContrast.grid(row=1, column=1)

            labelHue = tk.Label(sliderWin, text="Hue:", width=15)
            labelHue.grid(row=2, column=0)
            scaleHue = tk.Scale(sliderWin, from_=0, to=100, orient=tk.HORIZONTAL, command=sliderChoice, length=500,
                                width=25,
                                background='white', showvalue=False)
            scaleHue.set(0)
            scaleHue.grid(row=2, column=1)

            labelSaturation = tk.Label(sliderWin, text="Saturation:", width=15)
            labelSaturation.grid(row=3, column=0)
            scaleSaturation = tk.Scale(sliderWin, from_=0, to=200, orient=tk.HORIZONTAL, command=sliderChoice,
                                       length=500,
                                       width=25,
                                       background='white', showvalue=False)
            scaleSaturation.set(100)
            scaleSaturation.grid(row=3, column=1)

            labelBlurMod = tk.Label(sliderWin, text="Blur Mod:", width=15)
            labelBlurMod.grid(row=4, column=0)
            scaleBlurMod = tk.Scale(sliderWin, from_=0, to=30, orient=tk.HORIZONTAL, command=sliderChoice, length=500,
                                    width=25,
                                    background='white', showvalue=False)
            scaleBlurMod.set(0)
            scaleBlurMod.grid(row=4, column=1)

            labelBlurSigma = tk.Label(sliderWin, text="Blur Sigma:", width=15)
            labelBlurSigma.grid(row=5, column=0)
            scaleBlurSigma = tk.Scale(sliderWin, from_=0, to=30, orient=tk.HORIZONTAL, command=sliderChoice, length=500,
                                      width=25,
                                      background='white', showvalue=False)
            scaleBlurSigma.set(0)
            scaleBlurSigma.grid(row=5, column=1)

            labelSharpness = tk.Label(sliderWin, text="Sharpness:", width=15)
            labelSharpness.grid(row=6, column=0)
            scaleSharpness = tk.Scale(sliderWin, from_=0, to=60, orient=tk.HORIZONTAL, command=sliderChoice, length=500,
                                      width=25,
                                      background='white', showvalue=False)
            scaleSharpness.set(0)
            scaleSharpness.grid(row=6, column=1)

            labelSolarization = tk.Label(sliderWin, text="Solarization:", width=15)
            labelSolarization.grid(row=7, column=0)
            scaleSolarization = tk.Scale(sliderWin, from_=0, to=255, orient=tk.HORIZONTAL, command=sliderChoice,
                                         length=500,
                                         width=25,
                                         background='white', showvalue=False)
            scaleSolarization.set(255)
            scaleSolarization.grid(row=7, column=1)
            sliderWin.resizable(False, False)
            sliderChoice(0)
        else:
            tk.messagebox.showerror("Error", "You have not opened an image yet.")
    else:
        sliderWin.focus_force()


def crop():
    global Cropping
    Cropping = 1 if Cropping == 0 else 0


def helps():
    msg.showinfo("Help", '''Proiect AM (TMEAI) - Neleptcu Daniel-Andrei

    Open - Alegere fotografie de deschis in program.
    
    New Canvas - Introducere rezolutie pentru un canvas gol (alb)
    
    Save - Salvarea progresului intr-un fisier de format dorit.
    
    windowEfecte - Fereastra cu efecte fara modifier.
    
    EfecteSlider - Fereastra cu efecte cu modifier.
    
    Painting - Fereastra tool tip paint.

    ''')

    msg.showinfo("Help", '''Proiect AM (TMEAI) - Neleptcu Daniel-Andrei
        
        
    !!!!!!!!FEREASTRA windowEfecte!!!!!!!!
    
    
    Vignette- Centreaza luminozitatea imaginii (functioneaza bine pe imagini de dimensiuni reduse).
    
    Edge detection- Detecteaza marginile obiectelor din imagine folosind Canny, o idee mai precis decat Sobel.
    
    Grayscale- Transforma imaginea in tonuri de gri prin media celor 3 canale de culori.
    
    Vertical Flip- Oglindeste imaginea de-a lungul axei verticale.
    
    Horizontal Flip- Oglindeste imaginea de-a lungul axei orizontale.
    
    Sobel x- Aplica o masca de convolutie pentru a obtine un edge-detection pe axa Ox.
    
    Sobel y- Aplica o masca de convolutie pentru a obtine un edge-detection pe axa Oy.
    
    Sobel Full- Combina Sobel x si Sobel y pentru a obtine un edge-detection mai precis (subiectiv).
    
    Emboss- Aplica o masca de convolutie pentru a obtine un efect de reliefare.
    
    Sepia- Centreaza imaginea in jurul tonurilor de galben si maro.
    
    Sliders- Deschide o fereastra pentru a manipula efecte asupra imaginii prin intermediul sliderelor
    
    Remove BG- Tool extern pentru a scoate background-ul izoland obiectele. Ref: www.remove.bg
    
    Revert Fake Zoom - O incercare de manipulare a zoom-ului ( bibliotecile tkinter si cv2 nu prea interactioneaza )
     
    Crop - permite izolarea unei zone dreptunghiulare din poza.
    
    Undo- Posibilitatea de a anula o modificare anterioara.
    
    
    ''')

    msg.showinfo("Help", '''Proiect AM (TMEAI) - Neleptcu Daniel-Andrei
    
    
    !!!!!!!!FEREASTRA EfecteSlider!!!!!!!!
    
    
    Brightness- modifica luminozitatea imaginii, 0=imagine neagra, 512=imagine alba.
    
    Contrast- modifica diferenta de luminozitate / culoare intre regiunile imaginii.
    
    Hue- descrie nuanta pura a unei culori de baza (rosu, verde, albastru)
    
    Saturation- intensitatea culorilor de baza (rosu, verde, albastru)
    
    BlurMod- intensitatea kernelului de convolutie pentru blur-ul gaussian.
    
    Blur Sigma- deviatia standard pe axa Ox a kernelului de convolutie. Estompeaza conturul pe axa Ox.
    
    Sharpness- modifica claritatea/"ascutimea" detaliilor dintr-o imagine.
    
    Solarization- inversarea tonurilor de culoare.
    
    ''')

    msg.showinfo("Help", '''Proiect AM (TMEAI) - Neleptcu Daniel-Andrei
    
    
    !!!!!!!!FEREASTRA Painting!!!!!!!!
    
    
    Painting Brush- activeaza sau nu brush-ul pentru tool-ul de desenat
    TODO: Aplicarea izolata a efectelor din 'windowEfecte' si 'EfecteSlider' folosind un brush cand nu este bifat checkbox-ul.
    
    Selectie tip brush- permite selectia intre 3 tipuri de brush-uri:
    --Default- Brush default de desenat
    --Star- Brush in forma de "stea" de desenat
    --Spray- Brush de tip spray-can
    
    Color- permite selectia culorii cu un color-picker
    
    Density- alegerea densitatii punctelor/liniilor pentru brush-urile Spray/Star
    
    Star Angle- unghiul maxim la care se pot plasa liniile in cadrul brush-ului Star
    
    Brush Size- dimensiunea brush-ului.
    
    
    ''')


def newCanvas():
    global lastFunctionCall
    lastFunctionCall = 'newCanvas'

    heightVar = StringVar()
    widthVar = StringVar()
    pain = tk.Toplevel(root)
    pain.title("Introduceti Rezolutia")

    def setHeightWidth():
        global height, width, pixelMatrix, actualCvtCOLOR, wasMoved, sliderWinOpen
        initCanvas()
        height = int(heightVar.get() if heightVar.get() != "" else 0)
        width = int(widthVar.get() if widthVar.get() != "" else 0)
        if height > 0 and width > 0:
            pixelMatrix = np.ones((height, width, 3), dtype=np.uint8) * 255
            wasMoved = 0
            sliderWinOpen = 0
            ExportImgPreview()
            pain.destroy()
            canvas1.bind('<Button-1>', set_start)
            canvas1.bind('<ButtonRelease-1>', end_line)
        else:
            tk.messagebox.showerror("Error", "Enter a valid resolution")
            pain.focus_force()

    heightLabel = tk.Label(pain, text='Height', font=('calibre', 10, 'bold'))
    heightEntry = tk.Entry(pain, textvariable=heightVar, font=('calibre', 10, 'normal'))
    widthLabel = tk.Label(pain, text='Width', font=('calibre', 10, 'bold'))
    widthEntry = tk.Entry(pain, textvariable=widthVar, font=('calibre', 10, 'normal'))
    heightLabel.grid(row=0, column=0)
    heightEntry.grid(row=0, column=1)
    widthLabel.grid(row=1, column=0)
    widthEntry.grid(row=1, column=1)
    Button(pain, text="Ok", command=setHeightWidth).grid(row=2, column=0)


def choose_color():
    global color
    color = askcolor(color=color)[1]


def paint(event):
    global pixelMatrix, height, width, previewWidth, previewHeight, line_width
    global canvas1, color, choose_size_button, old_x, old_y
    global wasMoved
    wasMoved = 1
    xlb = (1400 - previewWidth) // 2
    xub = previewWidth + xlb
    ylb = (788 - previewHeight) // 2
    yub = previewHeight + ylb
    if xlb <= event.x <= xub and ylb <= event.y <= yub:
        if old_x and old_y:
            line_points.extend((event.x, event.y))
            line_id.append(canvas1.create_line(old_x, old_y, event.x, event.y,
                                               width=line_width, fill=color,
                                               capstyle=ROUND, smooth=TRUE, splinesteps=36))
            cv2.line(pixelMatrix,
                     (int((old_x - xlb) / previewWidth * width), int((old_y - ylb) / previewHeight * height)),
                     (int((event.x - xlb) / previewWidth * width), int((event.y - ylb) / previewHeight * height)),
                     ImageColor.getcolor(color, "RGB"), int(line_width * width / previewWidth))

        old_x = event.x
        old_y = event.y


def paintStar(event):
    global canvas1, previewWidth, previewHeight, line_width, color
    global wasMoved
    wasMoved = 1
    x, y = event.x, event.y

    xlb = (1400 - previewWidth) // 2
    xub = previewWidth + xlb
    ylb = (788 - previewHeight) // 2
    yub = previewHeight + ylb

    if xlb <= event.x <= xub and ylb <= event.y <= yub:
        for _ in range(density):
            angle = random.uniform(-angle_range / 2, angle_range / 2)

            x_offset = random.uniform(-5, 5)
            y_offset = random.uniform(-5, 5)

            x_end = x + line_width * math.cos(angle) + x_offset
            y_end = y + line_width * math.sin(angle) + y_offset

            line_points.extend((x, y))
            line_id.append(canvas1.create_line(x, y, x_end, y_end, fill=color))

            cv2.line(pixelMatrix, (int((x - xlb) / previewWidth * width), int((y - ylb) / previewHeight * height)),
                     (int((int(x_end) - xlb) / previewWidth * width), int((int(y_end) - ylb) / previewHeight * height)),
                     ImageColor.getcolor(color, "RGB"), 1)


def paintSpray(event):
    global canvas1, previewWidth, previewHeight, density, color
    global wasMoved
    wasMoved = 1

    x, y = event.x, event.y

    xlb = (1400 - previewWidth) // 2
    xub = previewWidth + xlb
    ylb = (788 - previewHeight) // 2
    yub = previewHeight + ylb

    if xlb <= event.x <= xub and ylb <= event.y <= yub:
        for _ in range(density):
            x_offset = random.uniform(-line_width // 2, line_width // 2)
            y_offset = random.uniform(-line_width // 2, line_width // 2)

            x_spray = x + x_offset
            y_spray = y + y_offset

            dot_size = 1
            line_points.extend((x_spray, y_spray))
            line_id.append(canvas1.create_oval(x_spray - dot_size, y_spray - dot_size,
                                               x_spray + dot_size, y_spray + dot_size, fill=color))

            cv2.circle(pixelMatrix,
                       (int((x_spray - xlb) / previewWidth * width), int((y_spray - ylb) / previewHeight * height)),
                       dot_size, ImageColor.getcolor(color, "RGB"), -1)


def deletePaint(pain):
    global lineWidthFlag, paintBrush
    lineWidthFlag = 0
    paintBrush.set(0)
    pain.destroy()


def painting():
    global paintBrush, color, old_x, old_y, choose_size_button, lineWidthFlag, options, tipBrush
    global densityScale, angleScale, validImage
    if validImage == 1:
        color = 'black'
        old_x = None
        old_y = None
        lineWidthFlag = 1
        pain = tk.Toplevel(root)
        pain.title("Paint")
        pain.iconbitmap("strudel_wla_icon.ico")
        pain.protocol("WM_DELETE_WINDOW", lambda: deletePaint(pain))

        C1 = Checkbutton(pain, text="Painting Brush", variable=paintBrush, onvalue=1, offvalue=0)
        C1.grid(row=0, column=0)

        options = [
            "Default",
            "Star",
            "Spray",
        ]
        C2 = OptionMenu(pain, tipBrush, *options)
        C2.config(font=("Arial", 16), width=7, height=2)
        C2.grid(row=1, column=0, padx=20)

        densityLabel = Label(pain, text='Density', font=('calibre', 10, 'bold'))
        densityLabel.grid(row=2, column=0, pady=(18, 0))
        densityScale = Scale(pain, from_=1, to=30, orient=HORIZONTAL, width=30, length=250)
        densityScale.grid(row=2, column=1)

        angleLabel = Label(pain, text='Star Angle', font=('calibre', 10, 'bold'))
        angleLabel.grid(row=3, column=0, pady=(18, 0))
        angleScale = Scale(pain, from_=0, to=360, orient=HORIZONTAL, width=30, length=250)
        angleScale.grid(row=3, column=1)

        color_button = Button(pain, text='color', command=choose_color, width=10, height=3)
        color_button.grid(row=1, column=1)

        sizeLabel = Label(pain, text='Brush Size', font=('calibre', 10, 'bold'))
        sizeLabel.grid(row=4, column=0, pady=(18, 0))
        choose_size_button = Scale(pain, from_=1, to=100, orient=HORIZONTAL, width=30, length=250)
        choose_size_button.grid(row=4, column=1)
    else:
        tk.messagebox.showerror("Error", "You have not opened an image yet.")


def revertFakeZoom():
    global x1, x2, y1, y2
    global pixelMatrix, pixelMatrixSlide, pixelMatrixStack
    global lastFunctionCall
    lastFunctionCall = 'revertFakeZoom'
    pixelMatrixSlide[y1:y2, x1:x2] = pixelMatrix
    pixelMatrix = copy.deepcopy(pixelMatrixSlide)
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def windowEfecte():
    global pixelMatrix, validImage
    if validImage == 1:
        efecte = tk.Toplevel(root)
        efecte.title("Efecte Statice")
        efecte.iconbitmap("strudel_wla_icon.ico")

        custom_font = font.Font(family="Helvetica", size=12, weight="bold")
        GradientButton = tk.Button(efecte, text="VIGNETTE", padx=10, pady=0, fg="white", bg="#358fb3",
                                   command=lambda: vignette(), width=15, height=2, font=custom_font, )
        GradientButton.grid(row=1, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        EdgeDetectionButton = tk.Button(efecte, text="EDGE DETECTION", padx=10, pady=0, fg="white", bg="#82bee8",
                                        command=lambda: edgify(), width=15, height=2, font=custom_font)

        EdgeDetectionButton.grid(row=2, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        GrayscaleButton = tk.Button(efecte, text="GRAYSCALE", padx=10, pady=0, fg="white", bg="#358fb3",
                                    command=lambda: grayscale(), width=15, height=2, font=custom_font)
        GrayscaleButton.grid(row=3, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        HorizontalFlipButton = tk.Button(efecte, text="VERTICAL FLIP", padx=10, pady=0, fg="white", bg="#82bee8",
                                         command=lambda: horizontal_flip(), width=15, height=2, font=custom_font)
        HorizontalFlipButton.grid(row=4, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        VerticalFlipButton = tk.Button(efecte, text="HORIZONTAL FLIP", padx=10, pady=0, fg="white", bg="#358fb3",
                                       command=lambda: vertical_flip(), width=15, height=2, font=custom_font)
        VerticalFlipButton.grid(row=5, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        sobelx = Button(efecte, text="SOBEL X",
                        command=lambda: sobel('x'), width=15, height=2, font=custom_font, padx=10,
                        pady=0, fg="white", bg="#82bee8")
        sobelx.grid(row=6, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        sobely = Button(efecte, text="SOBEL Y",
                        command=lambda: sobel('y'), width=15, height=2, font=custom_font, padx=10,
                        pady=0, fg="white", bg="#358fb3")
        sobely.grid(row=7, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        sobelf = Button(efecte, text="SOBEL FULL",
                        command=lambda: sobel('f'), width=15, height=2, font=custom_font, padx=10,
                        pady=0, fg="white", bg="#82bee8")
        sobelf.grid(row=1, column=1, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        slide_button = Button(efecte, text="EMBOSS", command=lambda: emboss(),
                              padx=10, pady=0, fg="white", bg="#358fb3", width=15, height=2, font=custom_font)
        slide_button.grid(row=2, column=1, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        slide_button = Button(efecte, text="SEPIA", command=lambda: sepia(),
                              padx=10, pady=0, fg="white", bg="#82bee8", width=15, height=2, font=custom_font)
        slide_button.grid(row=3, column=1, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        slide_button = Button(efecte, text="EFECTE DINAMICE", command=lambda: sliders(),
                              padx=10, pady=0, fg="white", bg="#358fb3", width=15, height=2, font=custom_font)
        slide_button.grid(row=4, column=1, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        undo_button = Button(efecte, text="REMOVE BG", command=lambda: removeBG(), padx=10,
                             pady=0, fg="white", bg="#82bee8", width=15, height=2, font=custom_font)
        undo_button.grid(row=5, column=1, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        removeBG_button = Button(efecte, text="REVERT FAKE ZOOM", command=lambda: revertFakeZoom(), padx=10,
                                 pady=0, fg="white", bg="#358fb3", width=15, height=2, font=custom_font)
        removeBG_button.grid(row=6, column=1, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)

        crop_button = Button(efecte, text="CROP", command=lambda: crop(), padx=10,
                             pady=0, fg="white", bg="#82bee8", width=15, height=2, font=custom_font)
        crop_button.grid(row=7, column=1, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0)
        revert_button = Button(efecte, text="UNDO", command=lambda: undo(), padx=2,
                               pady=0, fg="white", bg="#800000", width=34, height=2, font=custom_font)
        revert_button.grid(row=8, column=0, sticky=NW, padx=0, pady=0, ipadx=0, ipady=0, columnspan=2)
        efecte.resizable(False, False)
    else:
        tk.messagebox.showerror("Error", "You have not opened an image yet.")


def teste():
    global pixelMatrix
    kernel = [[-1, -4, -7, -10, -7, -4, -1],
              [-4, -8, -15, -22, -15, -8, -4],
              [-7, -15, 48, -48, -15, -7, -4],
              [-10, -22, -48, 214, -48, -22, -10],
              [-7, -15, -48, -48, -15, -7, -4],
              [-4, -8, -15, -22, -15, -8, -4],
              [-1, -4, -7, -10, -7, -4, -1]
              ]
    kernel = np.array(kernel)

    pixelMatrix = cv2.filter2D(pixelMatrix, -1, kernel)
    ExportImgPreview()


def draw_circle(canvas, x, y, radius, **kwargs):
    return canvas.create_oval(x - radius, y - radius, x + radius, y + radius, **kwargs)


def draw_line(event):
    global line_id, line_points, line_options, pixelMatrix, height, width, previewWidth, previewHeight
    global wasMoved
    wasMoved = 1
    xlb = (1400 - previewWidth) // 2
    xub = previewWidth + xlb
    ylb = (788 - previewHeight) // 2
    yub = previewHeight + ylb

    if xlb <= event.x <= xub and ylb <= event.y <= yub:
        line_points.extend((event.x, event.y))

        line_id.append(draw_circle(canvas1, event.x, event.y, 20, outline="yellow", fill="yellow"))

        cv2.circle(pixelMatrix,
                   (int((event.x - xlb) / previewWidth * width), int((event.y - ylb) / previewHeight * height)),
                   int(20 * (width / 1401)), (255, 0, 0), -1)


def cropping(event):
    global CropXInit, CropYInit, canvas1, oldRec, previewWidth, previewHeight
    global wasMoved
    wasMoved = 1
    xlb = (1400 - previewWidth) // 2
    xub = previewWidth + xlb
    ylb = (788 - previewHeight) // 2
    yub = previewHeight + ylb
    if xlb <= event.x <= xub and ylb <= event.y <= yub:
        global rec
        rec = canvas1.create_rectangle(CropXInit, CropYInit, event.x, event.y, outline='#F9FF00')
        canvas1.delete(oldRec)
        oldRec = rec


def set_start(event):
    global line_points, line_id, paintBrush, canvas1, line_width, choose_size_button, lineWidthFlag
    global tipBrush, densityScale, angleScale, density, angle_range, Cropping
    global CropXInit, CropYInit, sliderWinOpen, pixelMatrixStack, pixelMatrix, undoValue
    if sliderWinOpen == 1:
        temp = copy.deepcopy(pixelMatrix)
        pixelMatrixStack.append(temp)
        undoValue = undoValue + 1
    if Cropping == 0:
        if lineWidthFlag == 1:
            line_width = choose_size_button.get()
            density = int(densityScale.get())
            angle_range = int(angleScale.get()) * np.pi / 180
        line_id = []
        line_points.extend((event.x, event.y))

        if paintBrush.get() == 0:
            canvas1.bind('<B1-Motion>', draw_line)
        else:
            if tipBrush.get() == 'Default':
                canvas1.bind('<B1-Motion>', paint)
            elif tipBrush.get() == 'Star':
                canvas1.bind('<B1-Motion>', paintStar)
            elif tipBrush.get() == 'Spray':
                canvas1.bind('<B1-Motion>', paintSpray)
            else:
                canvas1.bind('<B1-Motion>', draw_line)
    else:
        CropXInit = event.x
        CropYInit = event.y
        canvas1.bind('<B1-Motion>', cropping)


def end_line(event):
    global line_points, line_id, pixelMatrix, pixelMatrixStack, paintBrush, old_x, old_y, rec
    global previewWidth, previewHeight, tipBrush, Cropping, CropXInit, CropYInit, width, height
    global x1, x2, y1, y2, pixelMatrixSlide, pixelMatrixSlideOriginal, lastFunctionCall, wasMoved
    global undoValue, sliderWinOpen
    if wasMoved == 1:
        if Cropping == 0:
            if line_id is not None:
                for i in line_id:
                    canvas1.delete(i)
            line_points.clear()
            line_id = None
            if tipBrush.get() != 'Default':
                xlb = (1400 - previewWidth) // 2
                xub = previewWidth + xlb
                ylb = (788 - previewHeight) // 2
                yub = previewHeight + ylb
                canvas1.create_rectangle(0, 0, 1400, ylb, fill="#000000", outline="#000000")
                canvas1.create_rectangle(0, 0, xlb, 788, fill="#000000", outline="#000000")
                canvas1.create_rectangle(0, yub + 1, 1400, 788, fill="#000000", outline="#000000")
                canvas1.create_rectangle(xub + 2, 0, 1400, 788, fill="#000000", outline="#000000")
            old_x = None
            old_y = None
            lastFunctionCall = 'drawing'
        else:
            Cropping = 0
            xlb = (1400 - previewWidth) // 2
            ylb = (788 - previewHeight) // 2
            # print("CropY coords: ", int((CropYInit - ylb) / previewHeight * height), int((event.y - ylb) / previewHeight * height))
            # print("CropX coords: ", int((CropXInit - xlb) / previewWidth * width), int((event.x - xlb) / previewWidth * width))
            y1 = min(int((CropYInit - ylb) / previewHeight * height), int((event.y - ylb) / previewHeight * height))
            y2 = max(int((CropYInit - ylb) / previewHeight * height), int((event.y - ylb) / previewHeight * height))
            x1 = min(int((CropXInit - xlb) / previewWidth * width), int((event.x - xlb) / previewWidth * width))
            x2 = max(int((CropXInit - xlb) / previewWidth * width), int((event.x - xlb) / previewWidth * width))
            pixelMatrixSlide = copy.deepcopy(pixelMatrix)
            pixelMatrix = pixelMatrix[y1:y2, x1:x2]
            canvas1.delete(rec)
            lastFunctionCall = 'cropping'

        if sliderWinOpen == 0:
            temp = copy.deepcopy(pixelMatrix)
            pixelMatrixStack.append(temp)
            pixelMatrixSlideOriginal = copy.deepcopy(pixelMatrix)
            undoValue = undoValue + 1

        ExportImgPreview()
        wasMoved = 0


def resizing():
    global lastFunctionCall, pixelMatrixStack, validResize, pixelMatrix
    if validResize == 1:
        imgDown = skimage.transform.rescale(pixelMatrix, resizeCoeff, anti_aliasing=True, channel_axis=2)
        pixelMatrix = (imgDown * 255).astype('uint8')
        pixelMatrixStack.append(pixelMatrix)
        lastFunctionCall = 'resize'
        ExportImgPreview()
        updateCoeff(0)
    validResize = 0


def updateCoeff(_):
    global resizeCoeff, height, width, resizeScale, validResize

    resizeCoeff = (resizeScale.get() + 100) / 20 - 5

    if resizeCoeff < 0:
        resizeCoeff = -1 / (resizeCoeff - 1)
    elif resizeCoeff == 0:
        resizeCoeff = 1
    elif resizeCoeff > 0:
        resizeCoeff = resizeCoeff + 1

    if height > 0 and width > 0:
        validResize = 1
        resizeHeight.set("Height: " + str(int(height * resizeCoeff)))
        resizeWidth.set("Width: " + str(int(width * resizeCoeff)))


def resizeImage():
    global resizeWidth, resizeHeight, resizeScale, height, width, validImage
    if validImage == 1:
        resizeWidth = StringVar()
        resizeWidth.set("Width: " + str(width))
        resizeHeight = StringVar()
        resizeHeight.set("Height: " + str(height))
        resizer = tk.Toplevel(root)
        resizer.title("Resize Image")
        resizer.geometry("505x170")
        resizer.resizable(False, False)
        root.iconbitmap("strudel_wla_icon.ico")
        tk.Label(resizer, justify=CENTER, textvariable=resizeWidth, font=("Helvetica", 15, "bold")).grid(row=0,
                                                                                                         column=0)
        tk.Label(resizer, justify=CENTER, textvariable=resizeHeight, font=("Helvetica", 15, "bold")).grid(row=0,
                                                                                                          column=1)
        resizeScale = tk.Scale(resizer, command=updateCoeff, orient=tk.HORIZONTAL, from_=-100, to=100, width=30,
                               length=500)
        resizeScale.grid(row=2, column=0, columnspan=2)
        tk.Button(resizer, text="Resize", command=resizing).grid(row=3, column=0, columnspan=2, pady=20)
    else:
        tk.messagebox.showerror("Error", "You have not opened an image yet.")


root = tk.Tk()
root.title("Proiect AM (TMEAI) Neleptcu Daniel-Andrei")
root.geometry("1404x813+-9+0")
root.configure(background="#000000")
root.iconbitmap("strudel_wla_icon.ico")
root.resizable(False, False)
mainmenu = Menu(root)
m1 = Menu(mainmenu, tearoff=0)
m1.add_command(label="Open", command=lambda: imgPreview())
m1.add_command(label="New Canvas", command=newCanvas)
m1.add_command(label="Save", command=saveimg)
m1.add_command(label="Efecte Statice", command=windowEfecte)
m1.add_command(label="Efecte Dinamice", command=sliders)
m1.add_command(label="Painting", command=painting)
m1.add_command(label="Reset", command=reset)
m1.add_command(label="Resize", command=resizeImage)

mainmenu.add_cascade(label="File", menu=m1)
root.config(menu=mainmenu)
root.protocol("WM_DELETE_WINDOW", lambda: deleteRoot())


def deleteRoot():
    global root
    stopMouse()
    root.destroy()


def startMouse():
    script_thread = threading.Thread(target=start_thread)
    script_thread.start()


def stopMouse():
    if script_process:
        script_process.terminate()


def on_arrow_key(event):
    global addKernelWin, varList
    current_widget = addKernelWin.focus_get()
    row, col = current_widget.grid_info()["row"], current_widget.grid_info()["column"]

    if event.keysym == "Up":
        row = max(1, row - 1)
    elif event.keysym == "Down":
        row = min(3, row + 1)
    elif event.keysym == "Left":
        col = max(0, col - 1)
    elif event.keysym == "Right":
        col = min(2, col + 1)

    next_widget = varList[row - 1][col]
    next_widget.focus_set()


def applyKernel():
    global pixelMatrix, pixelMatrixStack, oldSize, varList, lastFunctionCall
    kerFilter = []
    for i in range(oldSize):
        for j in range(oldSize):
            value = varList[i][j].get("1.0", 'end-1c')
            if value == "":
                value = 0
            else:
                value = eval(value)
            kerFilter.append(float(value))

    kerFilter = np.array(kerFilter).reshape((oldSize, oldSize))
    pixelMatrix = cv2.filter2D(pixelMatrix, -1, kerFilter)
    lastFunctionCall = 'applyCustomKernel'
    pixelMatrixStack.append(pixelMatrix)
    ExportImgPreview()


def changeSize(kernelSize):
    global varList, addKernelWin, oldSize
    print(oldSize)
    for i in range(oldSize):
        for j in range(oldSize):
            varList[i][j].destroy()
    if oldSize:
        varList[oldSize].destroy()

    varList = []
    for i in range(kernelSize):
        listRow = []
        for j in range(kernelSize):
            listRow.append(tk.Text(addKernelWin, height=1, width=4, padx=5, ))
            listRow[j].grid(row=i + 1, column=j)
        varList.append(listRow)

    varList.append(tk.Button(addKernelWin, text="Apply", command=lambda: applyKernel()))
    varList[kernelSize].grid(row=kernelSize + 1, column=0)
    oldSize = kernelSize
    addKernelWin.bind("<Up>", on_arrow_key)
    addKernelWin.bind("<Down>", on_arrow_key)
    addKernelWin.bind("<Left>", on_arrow_key)
    addKernelWin.bind("<Right>", on_arrow_key)


def executeFunction(value):
    global functionsBox, pixelMatrix, pixelMatrixStack, lastFunctionCall, validImage
    if validImage == 1:
        lastFunctionCall = 'execFunction'
        if value[-1] == '\n':
            functionName = value[:-1]
        else:
            functionName = value
        functionModule = importlib.import_module(functionName)
        pixelMatrix = functionModule.processImage(pixelMatrix)
        pixelMatrixStack.append(pixelMatrix)
        ExportImgPreview()
    else:
        tk.messagebox.showerror("Error", "You have not opened an image yet.")


def addKernel():
    global varList, addKernelWin
    addKernelWin = tk.Toplevel(root)
    addKernelWin.title("Add Your Kernel")
    addKernelWin.geometry("350x215")
    addKernelWin.iconbitmap("strudel_wla_icon.ico")
    varList = []
    tk.Button(addKernelWin, text="Size 3", command=lambda: changeSize(3)).grid(row=0, column=0)
    tk.Button(addKernelWin, text="Size 5", command=lambda: changeSize(5)).grid(row=0, column=1)
    tk.Button(addKernelWin, text="Size 7", command=lambda: changeSize(7)).grid(row=0, column=2)


def addFunction(fileName):
    def saveText():
        string = textbox.get("1.0", "end-1c")
        with open(string.splitlines()[0][1:] + ".py", "w") as file:
            file.write(string)
        flag = 1
        for index in range(m3.index("end") + 1):
            if m3.entrycget(index, "label") == string.splitlines()[0][1:]:
                flag = 0
                break

        if flag:
            m3.add_command(label=string.splitlines()[0][1:], command=lambda: executeFunction(string.splitlines()[0][1:]))
            with open("customFunctions.txt", "a") as fileCustoms:
                fileCustoms.write(string.splitlines()[0][1:] + "\n")
        tk.messagebox.showinfo("Info", "You have saved the Function.")
        funcWindow.destroy()

    funcWindow = tk.Toplevel(root)
    funcWindow.title("Custom Function")
    funcWindow.iconbitmap("strudel_wla_icon.ico")
    textbox = tk.Text(funcWindow, height=30, width=100)
    textbox.grid(row=0, column=0)
    if fileName != "":
        if fileName[-1] == '\n':
            fileName = fileName[:-1]
        with open(fileName + ".py", "r") as files:
            fileContent = files.read()
        textbox.insert(tk.END, fileContent)
    else:
        textbox.insert(tk.END,
                       "#fileName\n#the first lane contains the name of your custom function\n#do not change the name of the function below\n#import necessary packets\n#pixelMatrix is a cv2 image file\n\ndef processImage(pixelMatrix):\n\treturn pixelMatrix")
    textbutton = tk.Button(funcWindow, text="Save Function", command=saveText)
    textbutton.grid(row=1, column=0)


global functionsBox, delWindow


def deleteFunc():
    global functionsBox
    with open("customFunctions.txt", "r") as file:
        lines = file.readlines()

    text = functionsBox.get()
    if text[-1] == '\n':
        text = text[:-1]
    modified_lines = [lines for lines in lines if text not in lines]
    print(modified_lines)

    with open("customFunctions.txt", "w") as file:
        for lines in modified_lines:
            file.write(lines)
    os.remove(text + ".py")
    for index in range(m3.index("end")+1):
        if m3.entrycget(index, "label") == text:
            m3.delete(index)
            break
    functionsBox.configure(values=modified_lines)
    if len(modified_lines) > 0:
        functionsBox.current(0)
    tk.messagebox.showinfo("Info", "You have deleted the Function.")
    delWindow.destroy()
    modifyFunction()


def modifyFunction():
    global functionsBox, delWindow
    delWindow = tk.Toplevel(root)
    delWindow.title("Modify functions")
    delWindow.resizable(False, False)
    delWindow.geometry("290x200")
    delWindow.iconbitmap("strudel_wla_icon.ico")
    existingFunctions = []
    with open("customFunctions.txt", "r") as fileCustoms:
        for lines in fileCustoms:
            existingFunctions.append(lines)
    functionsBox = ttk.Combobox(delWindow, state="readonly", values=existingFunctions)
    functionsBox.place(x=50, y=50)
    tk.Button(delWindow, text="Delete", command=deleteFunc).place(x=50, y=100)
    tk.Button(delWindow, text="Modify", command=lambda: addFunction(functionsBox.get())).place(x=180, y=100)


m2 = Menu(mainmenu, tearoff=0)
m2.add_command(label="Help", command=helps)
m2.add_command(label="Start Mouse", command=startMouse)
m2.add_command(label="Stop Mouse", command=stopMouse)

mainmenu.add_cascade(label="Extra", menu=m2)
root.config(menu=mainmenu)

m3 = Menu(mainmenu, tearoff=0)
m3.add_command(label="Custom Kernel", command=addKernel)
m3.add_command(label="Add Function", command=lambda: addFunction(""))
m3.add_command(label="Modify Functions", command=modifyFunction)
with open("customFunctions.txt", "r") as fileCustom:
    for line in fileCustom:
        m3.add_command(label=line, command=lambda: executeFunction(line))
mainmenu.add_cascade(label="Custom", menu=m3)
root.config(menu=mainmenu)

canvas1 = tk.Canvas(root, height=786, width=1400, bg="#000000")

canvas1.grid(row=0, column=0)
oldSize = 0
line_id = []
line_points = []
line_options = {}
validImage = 0
Cropping = 0
oldRec = None
root.mainloop()
