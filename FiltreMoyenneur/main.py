import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# QUESTION 1
img_bgr = cv2.imread('lena_bruit.jpeg')
if img_bgr is not None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape
    print(f"Dimensions: {w} x {h} x {c}")

# QUESTION 2
def filtre_moyenneur(image, taille_noyau):
    h, w, c = image.shape
    image_filtree = np.zeros_like(image)
    offset = taille_noyau // 2
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            voisinage = image[i-offset : i+offset+1, j-offset : j+offset+1]
            image_filtree[i, j] = np.mean(voisinage, axis=(0, 1))
    return image_filtree.astype(np.uint8)

# QUESTION 3
tailles = [3, 5, 7, 11]
resultats = []
temps_calcul = []

for t in tailles:
    start = time.time()
    res = filtre_moyenneur(img_rgb, t)
    end = time.time()
    resultats.append(res)
    temps_calcul.append(end - start)

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Originale")
plt.axis('off')

for i, t in enumerate(tailles):
    plt.subplot(2, 3, i+2)
    plt.imshow(resultats[i])
    plt.title(f"Noyau {t}x{t}\nTemps: {temps_calcul[i]:.4f}s")
    plt.axis('off')

plt.tight_layout()
plt.show()

# QUESTION 4
filtre_opencv = cv2.blur(img_rgb, (5, 5))
mon_filtre_5x5 = resultats[1]
difference = cv2.absdiff(filtre_opencv, mon_filtre_5x5)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(mon_filtre_5x5); plt.title("Filtre Perso 5x5"); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(filtre_opencv); plt.title("OpenCV Blur 5x5"); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(difference); plt.title("Difference"); plt.axis('off')
plt.tight_layout()
plt.show()