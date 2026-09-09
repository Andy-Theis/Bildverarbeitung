import numpy as np
from PIL import Image #Pillow-Bibliothek für die Bildverarbeitung um ein Testbild zu erstellen und die Bilder zu speichern


def perona_malik_diffusion(img, iterations=20, delta_t=0.20, kappa=20):
    u = img.astype(np.float64)
    for _ in range(iterations):
        u_padded = np.pad(u, pad_width=1, mode="edge")

        grad_N = u_padded[:-2, 1:-1] - u
        grad_S = u_padded[2:, 1:-1] - u
        grad_O = u_padded[1:-1, 2:] - u
        grad_W = u_padded[1:-1, :-2] - u

        g_N = np.exp(-((grad_N / kappa) ** 2))
        g_S = np.exp(-((grad_S / kappa) ** 2))
        g_O = np.exp(-((grad_O / kappa) ** 2))
        g_W = np.exp(-((grad_W / kappa) ** 2))

        u += delta_t * (
            g_N * grad_N + g_S * grad_S + g_O * grad_O + g_W * grad_W
        )

    return np.clip(u, 0, 255).astype(np.uint8)


# 1. Synthetisches Testbild erstellen (Schwarz-Weiß-Quadrat)
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 200

# 2. Gaußsches Rauschen hinzufügen
noise = np.random.normal(0, 25, img.shape)
noisy_img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

# 3. Perona-Malik-Diffusion anwenden
denoised_img = perona_malik_diffusion(
    noisy_img, iterations=30, delta_t=0.20, kappa=15
)

# 4. Bilder als Datei speichern
Image.fromarray(noisy_img).save("verrauscht.png")
Image.fromarray(denoised_img).save("entrauscht.png")

print("Fertig! Bilder 'verrauscht.png' und 'entrauscht.png' wurden gespeichert.")