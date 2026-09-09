import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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


# 1. Eigenes Bild von der Festplatte laden & in Graustufen umwandeln
img_path = "mein_bild.jpg"  # Pfad zu deinem Bild anpassen
pil_img = Image.open(img_path).convert("L")
original_img = np.array(pil_img)

# 2. Gaußsches Rauschen zum eigenen Bild hinzufügen
noise = np.random.normal(0, 25, original_img.shape)
noisy_img = np.clip(original_img.astype(np.float64) + noise, 0, 255).astype(
    np.uint8
)

# 3. Perona-Malik-Diffusion anwenden
denoised_img = perona_malik_diffusion(
    noisy_img, iterations=30, delta_t=0.20, kappa=15
)

# 4. Darstellung mit Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Originalbild")
axes[0].axis("off")

axes[1].imshow(noisy_img, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Verrauscht")
axes[1].axis("off")

axes[2].imshow(denoised_img, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Perona-Malik Filter")
axes[2].axis("off")

plt.tight_layout()
plt.show()