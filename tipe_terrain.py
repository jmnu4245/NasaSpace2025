import numpy as np
import rasterio
from PIL import Image
import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


# Agua
COEF_AGUA = {
    "umbral": 0.01
}

# Arrozales
COEF_ARROZAL = {
    "vh_min": 0.02,
    "vh_max": 0.07,
    "vv_max": 0.09,
    "ratio_min": 0.4
}

# Vegetación
COEF_VEGETACION = {
    "ratio_min": 0.06
}

# Urbano/rocoso
COEF_URBANO = {
    "vv_min": 2
}

# Detección de quemado por diferencia temporal
COEF_QUEMADO_DIFERENCIA = {
    "vh_diff_min": -0.15,      # Disminución significativa en VH
    "vv_diff_min": -0.15,      # Disminución significativa en VV
    "ratio_diff_min": -0.25,   # Cambio negativo en el ratio
    "umbral_combinado": 0.6    # Umbral para la métrica combinada
}

def main():
    load_dotenv()
    
    # Cargar rutas de los GeoTIFF
    ruta_pre_incendio = os.getenv("RUTA_GEOTIFF_PRE")
    ruta_post_incendio = os.getenv("RUTA_GEOTIFF_POST")
    
    if not ruta_pre_incendio or not os.path.exists(ruta_pre_incendio):
        print("Error: RUTA_GEOTIFF_PRE no configurada o archivo no existe")
        return
    
    if not ruta_post_incendio or not os.path.exists(ruta_post_incendio):
        print("Error: RUTA_GEOTIFF_POST no configurada o archivo no existe")
        return
    
    nombre_archivo = "burn_analysis"
    carpeta_salida = f"output_{nombre_archivo}"
    carpeta_post = os.path.join(carpeta_salida, "post_incendio")
    carpeta_pre = os.path.join(carpeta_salida, "pre_incendio")
    
    os.makedirs(carpeta_post, exist_ok=True)
    os.makedirs(carpeta_pre, exist_ok=True)
    print(f"Carpeta de salida: {carpeta_salida}")
    print(f"  - POST-incendio: {carpeta_post}")
    print(f"  - PRE-incendio: {carpeta_pre}")
    
    # Leer imágenes pre y post incendio
    print(f"\nLeyendo imagen PRE-incendio: {ruta_pre_incendio}")
    vh_pre, vv_pre = leer_geotiff(ruta_pre_incendio)
    
    print(f"Leyendo imagen POST-incendio: {ruta_post_incendio}")
    vh_post, vv_post = leer_geotiff(ruta_post_incendio)
    
    # Verificar y ajustar dimensiones si no coinciden
    if vh_pre.shape != vh_post.shape or vv_pre.shape != vv_post.shape:
        print("Advertencia: Las dimensiones de las imágenes no coinciden. Recortando al tamaño más pequeño...")
        vh_pre, vv_pre, vh_post, vv_post = recortar_a_minimo(vh_pre, vv_pre, vh_post, vv_post)
        print(f"Nuevas dimensiones: {vh_pre.shape}")
    
    # Detectar áreas quemadas mediante diferencia temporal
    print("\nDetectando áreas quemadas por diferencia temporal...")
    quemado = detectar_quemado_diferencia(vh_pre, vv_pre, vh_post, vv_post)
    
    # Clasificar el terreno NO quemado usando la imagen post-incendio
    print("\nClasificando terreno NO quemado (imagen post-incendio)...")
    agua, arrozal, vegetacion, urbano, tierra = clasificar_terreno(vh_post, vv_post, quemado)
    
    # Clasificar el terreno SIN considerar quemado (imagen pre-incendio)
    print("\nClasificando terreno SIN quemado (imagen pre-incendio)...")
    agua_pre, arrozal_pre, vegetacion_pre, urbano_pre, tierra_pre = clasificar_terreno_sin_quemado(vh_pre, vv_pre)
    
    # Filtrar falsos positivos
    print("\nFiltrando capas con confianza...")
    agua       = filtrar_falsos_rapido(agua, vh_post, vv_post, confianza=0.9)
    arrozal    = filtrar_falsos_rapido(arrozal, vh_post, vv_post, confianza=0.5)
    vegetacion = filtrar_falsos_rapido(vegetacion, vh_post, vv_post, confianza=0.5)
    urbano     = filtrar_falsos_rapido(urbano, vh_post, vv_post, confianza=0.8)
    tierra     = filtrar_falsos_rapido(tierra, vh_post, vv_post, confianza=0.9)
    quemado    = filtrar_falsos_rapido(quemado, vh_post, vv_post, confianza=0.7)
    
    agua_pre       = filtrar_falsos_rapido(agua_pre, vh_pre, vv_pre, confianza=0.9)
    arrozal_pre    = filtrar_falsos_rapido(arrozal_pre, vh_pre, vv_pre, confianza=0.5)
    vegetacion_pre = filtrar_falsos_rapido(vegetacion_pre, vh_pre, vv_pre, confianza=0.5)
    urbano_pre     = filtrar_falsos_rapido(urbano_pre, vh_pre, vv_pre, confianza=0.8)
    tierra_pre     = filtrar_falsos_rapido(tierra_pre, vh_pre, vv_pre, confianza=0.9)

    # Guardar imágenes
    print("\nGuardando imágenes...")
    
    # Capas post-incendio (excluyendo quemado)
    Image.fromarray(agua, mode='L').save(os.path.join(carpeta_post, 'layer_water.png'))
    Image.fromarray(arrozal, mode='L').save(os.path.join(carpeta_post, 'layer_rice.png'))
    Image.fromarray(vegetacion, mode='L').save(os.path.join(carpeta_post, 'layer_vegetation.png'))
    Image.fromarray(urbano, mode='L').save(os.path.join(carpeta_post, 'layer_urban.png'))
    Image.fromarray(tierra, mode='L').save(os.path.join(carpeta_post, 'layer_soil.png'))
    Image.fromarray(quemado, mode='L').save(os.path.join(carpeta_post, 'layer_burned.png'))
    
    # Capas pre-incendio (sin considerar quemado)
    Image.fromarray(agua_pre, mode='L').save(os.path.join(carpeta_pre, 'layer_water.png'))
    Image.fromarray(arrozal_pre, mode='L').save(os.path.join(carpeta_pre, 'layer_rice.png'))
    Image.fromarray(vegetacion_pre, mode='L').save(os.path.join(carpeta_pre, 'layer_vegetation.png'))
    Image.fromarray(urbano_pre, mode='L').save(os.path.join(carpeta_pre, 'layer_urban.png'))
    Image.fromarray(tierra_pre, mode='L').save(os.path.join(carpeta_pre, 'layer_soil.png'))
    
    # Guardar también las imágenes de diferencia para análisis en carpeta post
    vh_diff = vh_post - vh_pre
    vv_diff = vv_post - vv_pre
    
    # Normalizar para visualización (0-255)
    vh_diff_norm = normalizar_para_visualizacion(vh_diff)
    vv_diff_norm = normalizar_para_visualizacion(vv_diff)
    
    Image.fromarray(vh_diff_norm, mode='L').save(os.path.join(carpeta_post, 'diff_vh.png'))
    Image.fromarray(vv_diff_norm, mode='L').save(os.path.join(carpeta_post, 'diff_vv.png'))
    
    print(f"\n¡Completado! Archivos guardados en: {carpeta_salida}/")


def detectar_quemado_diferencia(vh_pre, vv_pre, vh_post, vv_post):
    """
    Detecta áreas quemadas comparando imágenes SAR pre y post-incendio.
    Las áreas quemadas típicamente muestran:
    - Disminución en la retrodispersión VH y VV (superficie más lisa)
    - Cambio significativo en el ratio VH/VV
    """
    print("\n--- Análisis de diferencias temporales ---")
    
    # Calcular diferencias
    vh_diff = vh_post - vh_pre
    vv_diff = vv_post - vv_pre
    
    print(f"Diferencia VH - Min: {np.nanmin(vh_diff):.4f}, Max: {np.nanmax(vh_diff):.4f}, Media: {np.nanmean(vh_diff):.4f}")
    print(f"Diferencia VV - Min: {np.nanmin(vv_diff):.4f}, Max: {np.nanmax(vv_diff):.4f}, Media: {np.nanmean(vv_diff):.4f}")
    
    # Calcular ratios pre y post
    ratio_pre = np.where(vv_pre > 0.001, vh_pre / vv_pre, 0)
    ratio_post = np.where(vv_post > 0.001, vh_post / vv_post, 0)
    ratio_diff = ratio_post - ratio_pre
    
    print(f"Diferencia Ratio - Min: {np.nanmin(ratio_diff):.4f}, Max: {np.nanmax(ratio_diff):.4f}, Media: {np.nanmean(ratio_diff):.4f}")
    
    # Método 1: Detección basada en umbrales de diferencia
    quemado_vh = vh_diff < COEF_QUEMADO_DIFERENCIA["vh_diff_min"]
    quemado_vv = vv_diff < COEF_QUEMADO_DIFERENCIA["vv_diff_min"]
    quemado_ratio = ratio_diff < COEF_QUEMADO_DIFERENCIA["ratio_diff_min"]
    
    # Método 2: Métrica combinada normalizada (similar a dNBR para óptico)
    # Normalizamos las diferencias para crear un índice similar al dNBR
    epsilon = 1e-6
    
    # Índice de diferencia normalizada para VH
    ndi_vh = (vh_pre - vh_post) / (vh_pre + vh_post + epsilon)
    
    # Índice de diferencia normalizada para VV
    ndi_vv = (vv_pre - vv_post) / (vv_pre + vv_post + epsilon)
    
    # Métrica combinada
    metrica_quemado = (ndi_vh + ndi_vv) / 2
    
    print(f"Métrica quemado - Min: {np.nanmin(metrica_quemado):.4f}, Max: {np.nanmax(metrica_quemado):.4f}, Media: {np.nanmean(metrica_quemado):.4f}")
    print(f"Percentil 90 de métrica: {np.nanpercentile(metrica_quemado, 90):.4f}")
    print(f"Percentil 95 de métrica: {np.nanpercentile(metrica_quemado, 95):.4f}")
    
    # Combinar métodos: se considera quemado si cumple condiciones de diferencia O métrica alta
    quemado_mask = (
        ((quemado_vh & quemado_vv) | (quemado_vh & quemado_ratio)) |
        (metrica_quemado > COEF_QUEMADO_DIFERENCIA["umbral_combinado"])
    )
    
    # Excluir áreas de agua permanente (baja retrodispersión en ambas imágenes)
    agua_permanente = ((vh_pre < COEF_AGUA["umbral"]) & (vv_pre < COEF_AGUA["umbral"]) &
                       (vh_post < COEF_AGUA["umbral"]) & (vv_post < COEF_AGUA["umbral"]))
    
    quemado_mask = quemado_mask & ~agua_permanente
    
    quemado = np.zeros_like(vh_pre, dtype=np.uint8)
    quemado[quemado_mask] = 255
    
    print(f"\nÁrea quemada detectada: {np.sum(quemado_mask)} píxeles ({100*np.sum(quemado_mask)/quemado_mask.size:.2f}%)")
    
    return quemado


def clasificar_terreno(vh, vv, mascara_quemado):
    """
    Clasifica el terreno excluyendo las áreas ya identificadas como quemadas.
    """
    print("\n--- Clasificación de terreno NO quemado ---")
    
    ratio = np.where(vv > 0.001, vh / vv, 0)
    quemado_mask = mascara_quemado > 0
    
    # Agua
    agua = np.zeros_like(vh, dtype=np.uint8)
    agua_mask = (vh < COEF_AGUA["umbral"]) & (vv < COEF_AGUA["umbral"]) & ~quemado_mask
    agua[agua_mask] = 255
    print(f"Agua: {np.sum(agua_mask)} ({100*np.sum(agua_mask)/agua_mask.size:.2f}%)")

    # Arrozales
    arrozal = np.zeros_like(vh, dtype=np.uint8)
    arrozal_mask = (
        (vh >= COEF_ARROZAL["vh_min"]) & (vh < COEF_ARROZAL["vh_max"]) &
        (vv < COEF_ARROZAL["vv_max"]) &
        (ratio > COEF_ARROZAL["ratio_min"]) &
        ~quemado_mask
    )
    arrozal[arrozal_mask] = 255
    print(f"Arrozales: {np.sum(arrozal_mask)} ({100*np.sum(arrozal_mask)/arrozal_mask.size:.2f}%)")

    # Vegetación
    vegetacion = np.zeros_like(vh, dtype=np.uint8)
    zona_disponible = ~(agua_mask | arrozal_mask | quemado_mask)
    veg_mask = (ratio >= COEF_VEGETACION["ratio_min"]) & zona_disponible
    vegetacion[veg_mask] = 255
    print(f"Vegetación: {np.sum(veg_mask)} ({100*np.sum(veg_mask)/veg_mask.size:.2f}%)")

    # Urbano
    urbano = np.zeros_like(vh, dtype=np.uint8)
    zona_no_clasificada = zona_disponible & ~veg_mask
    urbano_mask = (vv >= COEF_URBANO["vv_min"]) & zona_no_clasificada
    urbano[urbano_mask] = 255
    print(f"Urbano: {np.sum(urbano_mask)} ({100*np.sum(urbano_mask)/urbano_mask.size:.2f}%)")

    # Tierra
    tierra = np.zeros_like(vh, dtype=np.uint8)
    tierra_mask = ~(agua_mask | arrozal_mask | veg_mask | urbano_mask | quemado_mask)
    tierra[tierra_mask] = 255
    print(f"Tierra: {np.sum(tierra_mask)} ({100*np.sum(tierra_mask)/tierra_mask.size:.2f}%)")

    return agua, arrozal, vegetacion, urbano, tierra


def clasificar_terreno_sin_quemado(vh, vv):
    """
    Clasifica el terreno SIN excluir áreas quemadas (para imagen pre-incendio).
    """
    print("\n--- Clasificación de terreno (sin filtro de quemado) ---")
    
    ratio = np.where(vv > 0.001, vh / vv, 0)
    
    # Agua
    agua = np.zeros_like(vh, dtype=np.uint8)
    agua_mask = (vh < COEF_AGUA["umbral"]) & (vv < COEF_AGUA["umbral"])
    agua[agua_mask] = 255
    print(f"Agua: {np.sum(agua_mask)} ({100*np.sum(agua_mask)/agua_mask.size:.2f}%)")

    # Arrozales
    arrozal = np.zeros_like(vh, dtype=np.uint8)
    arrozal_mask = (
        (vh >= COEF_ARROZAL["vh_min"]) & (vh < COEF_ARROZAL["vh_max"]) &
        (vv < COEF_ARROZAL["vv_max"]) &
        (ratio > COEF_ARROZAL["ratio_min"])
    )
    arrozal[arrozal_mask] = 255
    print(f"Arrozales: {np.sum(arrozal_mask)} ({100*np.sum(arrozal_mask)/arrozal_mask.size:.2f}%)")

    # Vegetación
    vegetacion = np.zeros_like(vh, dtype=np.uint8)
    zona_disponible = ~(agua_mask | arrozal_mask)
    veg_mask = (ratio >= COEF_VEGETACION["ratio_min"]) & zona_disponible
    vegetacion[veg_mask] = 255
    print(f"Vegetación: {np.sum(veg_mask)} ({100*np.sum(veg_mask)/veg_mask.size:.2f}%)")

    # Urbano
    urbano = np.zeros_like(vh, dtype=np.uint8)
    zona_no_clasificada = zona_disponible & ~veg_mask
    urbano_mask = (vv >= COEF_URBANO["vv_min"]) & zona_no_clasificada
    urbano[urbano_mask] = 255
    print(f"Urbano: {np.sum(urbano_mask)} ({100*np.sum(urbano_mask)/urbano_mask.size:.2f}%)")

    # Tierra
    tierra = np.zeros_like(vh, dtype=np.uint8)
    tierra_mask = ~(agua_mask | arrozal_mask | veg_mask | urbano_mask)
    tierra[tierra_mask] = 255
    print(f"Tierra: {np.sum(tierra_mask)} ({100*np.sum(tierra_mask)/tierra_mask.size:.2f}%)")

    return agua, arrozal, vegetacion, urbano, tierra


def normalizar_para_visualizacion(array):
    """
    Normaliza un array a rango 0-255 para visualización.
    """
    arr_min = np.nanmin(array)
    arr_max = np.nanmax(array)
    
    if arr_max - arr_min < 1e-6:
        return np.zeros_like(array, dtype=np.uint8)
    
    normalized = ((array - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return normalized


def filtrar_falsos_rapido(mask, vh, vv, n_clusters=3, contamination=0.05, confianza=0.7, max_samples=5000):
    """
    Filtra falsos positivos usando clustering y detección de anomalías.
    """
    mask_bin = mask > 0
    indices_mask = np.argwhere(mask_bin)
    coords_mask = np.column_stack([vh[mask_bin], vv[mask_bin]])
    
    if len(coords_mask) > max_samples:
        idx = np.random.choice(len(coords_mask), max_samples, replace=False)
        coords_mask_sub = coords_mask[idx]
        indices_sub = indices_mask[idx]
    else:
        coords_mask_sub = coords_mask
        indices_sub = indices_mask
    
    if len(coords_mask_sub) < n_clusters:
        return mask
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=5).fit(coords_mask_sub)
    labels = kmeans.labels_
    cluster_principal = np.bincount(labels).argmax()
    
    principal_coords = coords_mask_sub[labels == cluster_principal]
    if len(principal_coords) > 50:
        iso = IsolationForest(contamination=contamination, random_state=0).fit(principal_coords)
        pred = iso.predict(principal_coords)
    else:
        pred = np.ones(len(principal_coords))
    
    nueva_mask = mask.copy()
    j = 0
    for i, (x,y) in enumerate(indices_sub):
        if labels[i] == cluster_principal:
            if pred[j] == 1:
                nueva_mask[x,y] = 255
            else:
                if np.random.rand() > confianza:
                    nueva_mask[x,y] = 0
            j += 1
    
    indices_fuera = np.argwhere(~mask_bin)
    coords_fuera = np.column_stack([vh[~mask_bin], vv[~mask_bin]])
    if len(coords_fuera) > 0:
        centroide = kmeans.cluster_centers_[cluster_principal]
        dist = np.linalg.norm(coords_fuera - centroide, axis=1)
        umbral = np.percentile(dist, int(10*(1-confianza)))
        for i, d in enumerate(dist):
            if d <= umbral:
                x, y = indices_fuera[i]
                nueva_mask[x,y] = 255
    
    return nueva_mask


def leer_geotiff(ruta_archivo):
    """
    Lee un archivo GeoTIFF y extrae las bandas VH y VV.
    """
    with rasterio.open(ruta_archivo) as src:
        vh = src.read(1)
        vv = src.read(2)
        return vh, vv


def recortar_a_minimo(vh_pre, vv_pre, vh_post, vv_post):
    """
    Recorta todas las imágenes al tamaño mínimo común.
    """
    # Encontrar dimensiones mínimas
    min_rows = min(vh_pre.shape[0], vh_post.shape[0])
    min_cols = min(vh_pre.shape[1], vh_post.shape[1])
    
    print(f"Dimensiones originales:")
    print(f"  PRE:  {vh_pre.shape}")
    print(f"  POST: {vh_post.shape}")
    print(f"Recortando a: ({min_rows}, {min_cols})")
    
    # Recortar todas las imágenes
    vh_pre_crop = vh_pre[:min_rows, :min_cols]
    vv_pre_crop = vv_pre[:min_rows, :min_cols]
    vh_post_crop = vh_post[:min_rows, :min_cols]
    vv_post_crop = vv_post[:min_rows, :min_cols]
    
    return vh_pre_crop, vv_pre_crop, vh_post_crop, vv_post_crop


if __name__ == '__main__':
    main()