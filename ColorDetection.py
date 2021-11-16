import cv2 
import numpy as np
import matplotlib.pyplot as plt
from Functions import LicensePlateDetector
from Functions import Color

def ColorDetection(path):
    """ YOLOv3 - BoundingBox Carro"""
    lpd = LicensePlateDetector(
    pth_weights='car.weights', 
    pth_cfg='yolov3-testing.cfg', 
    pth_classes='classes.txt')
    
    lpd.detect(path)
    original = cv2.imread(path)
    cv2.imshow("figure", original)
    
    x, y, w, h, roi = lpd.crop_plate()
    cropada = (lpd.roi_image)
    
    rect_final = (x, y, w, h)
    img_origG = Color.run_grabcut(cropada, rect_final)
    
    """ Função GrabCut para retirar o barckground """
    img_origG[np.where((img_origG==[0,0,0]).all(axis=2))] = [0,254,0]
    # cv2.imwrite("grabcut.jpg", img_origG)
    
    """ Histograma de cor e KMeans - Dominant Color """
    img = cv2.cvtColor(img_origG, cv2.COLOR_RGB2BGR)
    
    pixel = np.float32(img.reshape(-1, 3))
    pixels = np.delete(pixel, np.where((pixel==[0,254,0]).all(axis=1)), 0)
    
    # average = img.mean(axis=0).mean(axis=0)
    average = pixels.mean(axis=0)
    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 5, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    
    avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)
    
    indices = np.argsort(counts)[::-1]   
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    rows = np.int_(img.shape[0]*freqs)
    
    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
        
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6,3))
    ax0.imshow(avg_patch)
    ax0.set_title('Average Color')
    ax0.axis('off')
    ax1.imshow(dom_patch)
    ax1.set_title('Dominant Colors')
    ax1.axis('off')
    
    """ Repasse da cor dominante em HSV """
    hsvF = Color.rgb_to_hsv(dominant[0], dominant[1], dominant[2])
    hsv = [hsvF[0], hsvF[1]/100, hsvF[2]/100]
    if (hsv[0] > 0 and hsv[0] < 360 and hsv[1] > 0 and hsv[1] < 1 and hsv[2] < 0.1):
        FinalColor = "Black"
    elif (hsv[0] > 0 and hsv[0] < 360 and hsv[1] < 0.15 and hsv[2] > 0.65):
        FinalColor = "White"
    elif (hsv[0] > 0 and hsv[0] < 360 and hsv[1] < 0.15 and hsv[2] > 0.1 and hsv[2] < 0.65):
        FinalColor = "Gray"
    elif ((hsv[0] < 11 or hsv[0] > 330) and hsv[1] > 0.36 and hsv[2] > 0.1):
        FinalColor = "Red"
    elif(hsv[0] > 310 and hsv[0] <= 330 and hsv[1] > 0.15 and hsv[2] > 0.1):
         FinalColor = "Pink"
    elif(hsv[0] > 11 and hsv[0] < 45 and hsv[1] > 0.15 and hsv[2] > 0.75):
        FinalColor = "Orange"
    elif(hsv[0] > 11 and hsv[0] < 45 and hsv[1] > 0.15 and hsv[2] > 0.1 and hsv[2] < 0.75):
        FinalColor = "Brown"
    elif(hsv[0] > 45 and hsv[0] < 64 and hsv[1] > 0.15 and hsv[2] > 0.1):
        FinalColor = "Yellow"
    elif(hsv[0] > 64 and hsv[0] < 165 and hsv[1] > 0.15 and hsv[2] > 0.1):
        FinalColor = "Green"
    elif(hsv[0] >= 165 and hsv[0] < 255 and hsv[1] > 0.15 and hsv[2] > 0.1):
        FinalColor = "Blue"
    elif(hsv[0] > 255 and hsv[0] < 310 and hsv[1] > 0.5 and hsv[2] > 0.1):
        FinalColor = "Purple"
    elif(hsv[0] > 255 and hsv[0] < 310 and hsv[1] > 0.15 and hsv[1] < 0.35 and hsv[2] > 0.1):
        FinalColor = "LightPurple"
        
    print("\nColor:", FinalColor, "| RGB Value:", dominant.astype(int))
    # print("\nRGB", dominant)
    # print("\nHSV", hsvF)
    
    return FinalColor
