import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Entrada/salida
INPUT_PATH = "resources/DEM.jpg"
OUT_SMOOTH = "grises_minecraft.png"

# Escala de colores (altura -> RGB)
color_scale = {
    8000: (255, 255, 255),
    7000: (242, 242, 242),
    6000: (229, 229, 229),
    5500: (73, 56, 17),
    5000: (94, 76, 38),
    4500: (114, 96, 56),
    4000: (135, 114, 76),
    3500: (153, 135, 96),
    3000: (173, 155, 117),
    2500: (193, 175, 137),
    2000: (214, 196, 158),
    1500: (234, 216, 175),
    1000: (252, 237, 191),
    900:  (170, 221, 160),
    800:  (165, 214, 155),
    700:  (150, 206, 142),
    600:  (132, 193, 122),
    500:  (122, 186, 112),
    400:  (114, 178, 102),
    300:  (94, 163, 84),
    200:  (76, 147, 63),
    100:  (61, 135, 61),
    75:   (53, 124, 58),
    50:   (45, 114, 45),
    25:   (38, 104, 33),
    10:   (30, 94, 20),
    0.00001: (22, 84, 7),
    0:    (15, 15, 140)
}

heights = np.array(list(color_scale.keys()), dtype=np.float32)
colors = np.array(list(color_scale.values()), dtype=np.int16)

def rgb_to_height(img_arr):
    h, w, _ = img_arr.shape
    pixels = img_arr.reshape(-1, 3).astype(np.int16)
    diffs = np.linalg.norm(pixels[:, None, :] - colors[None, :, :], axis=2)
    idxs = np.argmin(diffs, axis=1)
    return heights[idxs].reshape(h, w)

def scale_to_minecraft(heightmap, sea_level=62, vertical_scale=5, mc_max=255):
    """
    Convierte metros -> bloques con factor vertical_scale (m/bloque).
    Ej: vertical_scale=5 -> 75 m = 15 bloques.
    """
    blocks = np.rint(heightmap / vertical_scale).astype(np.int32)
    gray = np.clip(sea_level + blocks, 0, mc_max).astype(np.uint8)
    return gray

def main():
    img = Image.open(INPUT_PATH).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    # Colores -> alturas en metros
    heightmap = rgb_to_height(arr)

    # Suavizado
    smooth = gaussian_filter(heightmap, sigma=3)

    # Escalar a Minecraft con factor vertical
    gray_smooth = scale_to_minecraft(smooth, sea_level=62, vertical_scale=5)
    Image.fromarray(gray_smooth, mode='L').save(OUT_SMOOTH)

    print(f"[OK] Heightmap guardado en {OUT_SMOOTH} con escala vertical 1 bloque = 5 m")

if __name__ == "__main__":
    main()
