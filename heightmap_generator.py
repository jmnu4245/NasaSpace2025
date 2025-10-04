from PIL import Image
import numpy as np

#Diccionario de colores para diferentes tipos de terreno
path = "resources/heightmap.png"
color_to_height = map_rgb = {
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

if __name__ == "__main__":
   # Cargar la imagen PNG
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    # Crear una matriz para las alturas
    height_map = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.float32)

    # Recorremos los píxeles y asignamos alturas
    for color, h in color_to_height.items():
        mask = np.all(arr == color, axis=-1)
        height_map[mask] = h

    # Normalizar a 0-255 para Minecraft
    min_h, max_h = np.min(height_map), np.max(height_map)
    height_norm = ((height_map - min_h) / (max_h - min_h) * 255).astype(np.uint8)

    # Guardar el heightmap en escala de grises
    out = Image.fromarray(height_norm, mode="L")
    out.save("heightmap.png")