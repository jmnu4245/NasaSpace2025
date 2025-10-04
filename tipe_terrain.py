import numpy as np
import rasterio
from PIL import Image
import os
from scipy.ndimage import convolve
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Agua
COEF_AGUA = {
    "umbral": 0.02
}

# Arrozales
COEF_ARROZAL = {
    "vh_min": 0.02,
    "vh_max": 0.07,
    "vv_max": 0.09,
    "ratio_min": 0.5
}

# Vegetación
COEF_VEGETACION = {
    "ratio_min": 0.3
}

# Urbano/rocoso
COEF_URBANO = {
    "vv_min": 2
}
def main():
    # Configuración
    load_dotenv()
    ruta_geotiff = os.getenv("RUTA_GEOTIFF2")
    
    if not ruta_geotiff or not os.path.exists(ruta_geotiff):
        print("Error: RUTA_GEOTIFF no configurada o archivo no existe")
        return
    
    # Obtener nombre del archivo sin extensión
    nombre_archivo = os.path.splitext(os.path.basename(ruta_geotiff))[0]
    
    # Crear carpeta de salida
    carpeta_salida = f"output_{nombre_archivo}"
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"Carpeta de salida: {carpeta_salida}")
    
    # Leer datos
    print(f"Leyendo {ruta_geotiff}...")
    vh, vv, altura = leer_geotiff(ruta_geotiff)
    
    # Clasificar terreno en capas
    print("Clasificando terreno en capas...")
    agua, arrozal, vegetacion, urbano, tierra = clasificar_terreno(vh, vv)
    
    print("Filtrando falsos positivos y recuperando falsos negativos...")
    agua       = filtrar_falsos(agua, vh, vv, altura)
    arrozal    = filtrar_falsos(arrozal, vh, vv, altura)
    vegetacion = filtrar_falsos(vegetacion, vh, vv, altura)
    urbano     = filtrar_falsos(urbano, vh, vv, altura)
    tierra     = filtrar_falsos(tierra, vh, vv, altura)

    # Normalizar altura
    print("Procesando altura...")
    altura_16bit = normalizar_altura(altura, bits=16)
    


    # Guardar PNGs en la carpeta
    print("Guardando imágenes...")
    Image.fromarray(altura_16bit, mode='I;16').save(os.path.join(carpeta_salida, 'layer_height.png'))
    Image.fromarray(agua, mode='L').save(os.path.join(carpeta_salida, 'layer_water.png'))
    Image.fromarray(arrozal, mode='L').save(os.path.join(carpeta_salida, 'layer_rice.png'))
    Image.fromarray(vegetacion, mode='L').save(os.path.join(carpeta_salida, 'layer_vegetation.png'))
    Image.fromarray(urbano, mode='L').save(os.path.join(carpeta_salida, 'layer_urban.png'))
    Image.fromarray(tierra, mode='L').save(os.path.join(carpeta_salida, 'layer_soil.png'))
    
    print(f"\n¡Completado! Archivos guardados en: {carpeta_salida}/")

def filtrar_falsos(mask, vh, vv, altura, n_clusters=3, contamination=0.05):
    """
    Filtra falsos positivos y añade falsos negativos usando KMeans + IsolationForest.
    - mask: matriz binaria de la capa (0/255)
    - vh, vv, altura: matrices con los valores de cada píxel
    - n_clusters: número de clusters para KMeans
    - contamination: proporción de outliers para IsolationForest
    Retorna la máscara filtrada (0/255)
    """
    mask_bin = mask > 0
    coords_mask = np.column_stack([vh[mask_bin], vv[mask_bin], altura[mask_bin]])
    
    if len(coords_mask) < n_clusters:
        return mask  # demasiado pocos puntos
    
    # ---------------------------
    # 1️⃣ Filtrar falsos positivos
    # ---------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords_mask)
    labels = kmeans.labels_
    cluster_principal = np.bincount(labels).argmax()
    
    # IsolationForest dentro del cluster principal
    principal_coords = coords_mask[labels == cluster_principal]
    iso = IsolationForest(contamination=contamination, random_state=0).fit(principal_coords)
    pred = iso.predict(principal_coords)  # 1=normal, -1=outlier
    
    nueva_mask = np.zeros_like(mask, dtype=np.uint8)
    indices = np.argwhere(mask_bin)
    j = 0
    for i, (x,y) in enumerate(indices):
        if labels[i] == cluster_principal:
            if pred[j] == 1:  # mantener solo los normales
                nueva_mask[x,y] = 255
            j += 1

    # ---------------------------
    # 2️⃣ Añadir falsos negativos (píxeles afuera que parecen correctos)
    # ---------------------------
    fuera_mask = np.column_stack([vh[~mask_bin], vv[~mask_bin], altura[~mask_bin]])
    if len(fuera_mask) > 0:
        # comparar distancia euclídea al centroide del cluster principal
        centroide = kmeans.cluster_centers_[cluster_principal]
        dist = np.linalg.norm(fuera_mask - centroide, axis=1)
        umbral = np.percentile(dist, 10)  # añadir los más cercanos
        indices_fuera = np.argwhere(~mask_bin)
        for i, d in enumerate(dist):
            if d <= umbral:
                x, y = indices_fuera[i]
                nueva_mask[x,y] = 255
    
    return nueva_mask

def filtrar_falsos_positivos(matriz, umbral_vecinos=3):
    """
    Aplica convolución 2D para eliminar falsos positivos aislados en una máscara binaria.
    - matriz: np.ndarray binaria (0/255).
    - umbral_vecinos: número mínimo de vecinos necesarios para conservar el píxel.
    Retorna la matriz filtrada.
    """
    # Kernel 3x3 que cuenta los vecinos (excepto el centro)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)

    # Convertimos la matriz a 0/1
    binaria = (matriz > 0).astype(np.uint8)

    # Conteo de vecinos con convolución 2D
    vecinos = convolve(binaria, kernel, mode='constant', cval=0)

    # Decisión: mantener píxel solo si tiene suficientes vecinos
    filtrada = np.where((binaria == 1) & (vecinos >= umbral_vecinos), 255, 0).astype(np.uint8)

    return filtrada

def leer_geotiff(ruta_archivo):
    """Lee GeoTIFF con bandas VH, VV y altura"""
    with rasterio.open(ruta_archivo) as src:
        vh = src.read(1)
        vv = src.read(2)
        altura = src.read(3)
        return vh, vv, altura

def clasificar_terreno(vh, vv):
    # Estadísticas
    print("\n--- Estadísticas de datos SAR ---")
    print(f"VH - Min: {np.nanmin(vh):.2f}, Max: {np.nanmax(vh):.2f}, Media: {np.nanmean(vh):.2f}")
    print(f"VV - Min: {np.nanmin(vv):.2f}, Max: {np.nanmax(vv):.2f}, Media: {np.nanmean(vv):.2f}")

    diferencia = vv - vh
    print(f"VV-VH - Min: {np.nanmin(diferencia):.2f}, Max: {np.nanmax(diferencia):.2f}, Media: {np.nanmean(diferencia):.2f}")

    # Agua
    agua = np.zeros_like(vh, dtype=np.uint8)
    agua_mask = (vh < COEF_AGUA["umbral"]) & (vv < COEF_AGUA["umbral"])
    agua[agua_mask] = 255
    print(f"Agua: {np.sum(agua_mask)} ({100*np.sum(agua_mask)/agua_mask.size:.1f}%)")

    # Arrozales
    arrozal = np.zeros_like(vh, dtype=np.uint8)
    ratio = np.where(vv > 0.001, vh / vv, 0)
    arrozal_mask = (
        (vh >= COEF_ARROZAL["vh_min"]) & (vh < COEF_ARROZAL["vh_max"]) &
        (vv < COEF_ARROZAL["vv_max"]) &
        (ratio > COEF_ARROZAL["ratio_min"])
    )
    arrozal[arrozal_mask] = 255
    print(f"Arrozales: {np.sum(arrozal_mask)} ({100*np.sum(arrozal_mask)/arrozal_mask.size:.1f}%)")

    # Vegetación
    vegetacion = np.zeros_like(vh, dtype=np.uint8)
    tierra_seca_mask = ~(agua_mask | arrozal_mask)
    veg_mask = (ratio >= COEF_VEGETACION["ratio_min"]) & tierra_seca_mask
    vegetacion[veg_mask] = 255
    print(f"Vegetación: {np.sum(veg_mask)} ({100*np.sum(veg_mask)/veg_mask.size:.1f}%)")

    # Urbano
    urbano = np.zeros_like(vh, dtype=np.uint8)
    zona_no_clasificada = tierra_seca_mask & ~veg_mask
    urbano_mask = (vv >= COEF_URBANO["vv_min"]) & zona_no_clasificada
    urbano[urbano_mask] = 255
    print(f"Urbano: {np.sum(urbano_mask)} ({100*np.sum(urbano_mask)/urbano_mask.size:.1f}%)")

    # Tierra
    tierra = np.zeros_like(vh, dtype=np.uint8)
    tierra_mask = ~(agua_mask | arrozal_mask | veg_mask | urbano_mask)
    tierra[tierra_mask] = 255
    print(f"Tierra: {np.sum(tierra_mask)} ({100*np.sum(tierra_mask)/tierra_mask.size:.1f}%)")

    return agua, arrozal, vegetacion, urbano, tierra

def normalizar_altura(altura, bits=16):
    """Normaliza altura a rango 0-65535 (16-bit) o 0-255 (8-bit)"""
    altura_norm = (altura - np.nanmin(altura)) / (np.nanmax(altura) - np.nanmin(altura))
    if bits == 16:
        altura_norm = (altura_norm * 65535).astype(np.uint16)
    else:
        altura_norm = (altura_norm * 255).astype(np.uint8)
    return altura_norm



if __name__ == '__main__':
    main()