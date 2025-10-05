import numpy as np
import rasterio
from PIL import Image
import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


# Agua
COEF_AGUA = {
    "umbral": 0.05
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
    "ratio_min": 2
}

# Urbano/rocoso
COEF_URBANO = {
    "vv_min": 2
}

# Zona quemada
COEF_QUEMADO = {
    "vh_max": -2.0,       # Valores muy bajos en dB
    "vv_max": -3.0,
    "ratio_max": 0.20     # Ratio muy bajo indica superficie lisa/quemada
}


def main():
    load_dotenv()
    ruta_geotiff = os.getenv("RUTA_GEOTIFF1")
    
    if not ruta_geotiff or not os.path.exists(ruta_geotiff):
        print("Error: RUTA_GEOTIFF no configurada o archivo no existe")
        return
    
    nombre_archivo = os.path.splitext(os.path.basename(ruta_geotiff))[0]
    carpeta_salida = f"output_{nombre_archivo}"
    os.makedirs(carpeta_salida, exist_ok=True)
    print(f"Carpeta de salida: {carpeta_salida}")
    
    print(f"Leyendo {ruta_geotiff}...")
    vh, vv = leer_geotiff(ruta_geotiff)
    
    print("Clasificando terreno en capas ...")
    agua, arrozal, vegetacion, urbano, tierra,quemado = clasificar_terreno_quemado(vh, vv)
    
    
    print("Filtrando capas con confianza...")
    agua       = filtrar_falsos_rapido(agua, vh, vv, confianza=0.9)
    arrozal    = filtrar_falsos_rapido(arrozal, vh, vv, confianza=0.5)
    vegetacion = filtrar_falsos_rapido(vegetacion, vh, vv, confianza=0.5)
    urbano     = filtrar_falsos_rapido(urbano, vh, vv, confianza=0.8)
    tierra     = filtrar_falsos_rapido(tierra, vh, vv, confianza=0.9)
    quemado    = filtrar_falsos_rapido(quemado, vh, vv, confianza=0.7)

    print("Guardando imágenes...")
    Image.fromarray(agua, mode='L').save(os.path.join(carpeta_salida, 'layer_water.png'))
    Image.fromarray(arrozal, mode='L').save(os.path.join(carpeta_salida, 'layer_rice.png'))
    Image.fromarray(vegetacion, mode='L').save(os.path.join(carpeta_salida, 'layer_vegetation.png'))
    Image.fromarray(urbano, mode='L').save(os.path.join(carpeta_salida, 'layer_urban.png'))
    Image.fromarray(tierra, mode='L').save(os.path.join(carpeta_salida, 'layer_soil.png'))
    Image.fromarray(quemado, mode='L').save(os.path.join(carpeta_salida, 'layer_burned.png'))
    
    print(f"\n¡Completado! Archivos guardados en: {carpeta_salida}/")


def filtrar_falsos_rapido(mask, vh, vv, n_clusters=3, contamination=0.05, confianza=0.7, max_samples=5000):
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
    with rasterio.open(ruta_archivo) as src:
        vh = src.read(1)
        vv = src.read(2)
        return vh, vv


def clasificar_terreno(vh, vv):
    print("\n--- Estadísticas de datos SAR ---")
    print(f"VH - Min: {np.nanmin(vh):.2f}, Max: {np.nanmax(vh):.2f}, Media: {np.nanmean(vh):.2f}")
    print(f"VV - Min: {np.nanmin(vv):.2f}, Max: {np.nanmax(vv):.2f}, Media: {np.nanmean(vv):.2f}")

    diferencia = vv - vh
    print(f"VV-VH - Min: {np.nanmin(diferencia):.2f}, Max: {np.nanmax(diferencia):.2f}, Media: {np.nanmean(diferencia):.2f}")

    # Agua
    agua = np.zeros_like(vh, dtype=np.uint8)
    agua_mask = (vh < COEF_AGUA["umbral"]) & (vv < COEF_AGUA["umbral"])
    agua[agua_mask] = 255
    print(f"\nAgua: {np.sum(agua_mask)} ({100*np.sum(agua_mask)/agua_mask.size:.1f}%)")

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

    # Zona quemada (valores absolutos muy bajos + ratio bajo)
    quemado = np.zeros_like(vh, dtype=np.uint8)
    tierra_seca_mask = ~(agua_mask | arrozal_mask)
    quemado_mask = (
        (vh < COEF_QUEMADO["vh_max"]) &
        (vv < COEF_QUEMADO["vv_max"]) &
        (ratio < COEF_QUEMADO["ratio_max"]) &
        tierra_seca_mask
    )
    quemado[quemado_mask] = 255
    print(f"Zona quemada: {np.sum(quemado_mask)} ({100*np.sum(quemado_mask)/quemado_mask.size:.1f}%)")

    # Vegetación (excluyendo quemado)
    vegetacion = np.zeros_like(vh, dtype=np.uint8)
    zona_no_quemada = tierra_seca_mask & ~quemado_mask
    veg_mask = (ratio >= COEF_VEGETACION["ratio_min"]) & zona_no_quemada
    vegetacion[veg_mask] = 255
    print(f"Vegetación: {np.sum(veg_mask)} ({100*np.sum(veg_mask)/veg_mask.size:.1f}%)")

    # Urbano
    urbano = np.zeros_like(vh, dtype=np.uint8)
    zona_no_clasificada = zona_no_quemada & ~veg_mask
    urbano_mask = (vv >= COEF_URBANO["vv_min"]) & zona_no_clasificada
    urbano[urbano_mask] = 255
    print(f"Urbano: {np.sum(urbano_mask)} ({100*np.sum(urbano_mask)/urbano_mask.size:.1f}%)")

    # Tierra
    tierra = np.zeros_like(vh, dtype=np.uint8)
    tierra_mask = ~(agua_mask | arrozal_mask | quemado_mask | veg_mask | urbano_mask)
    tierra[tierra_mask] = 255
    print(f"Tierra: {np.sum(tierra_mask)} ({100*np.sum(tierra_mask)/tierra_mask.size:.1f}%)")

    return agua, arrozal, vegetacion, urbano, tierra, quemado

def clasificar_terreno_quemado(vh, vv):
    print("\n--- Clasificación con detección de quemado ---")
    
    ratio = np.where(vv > 0.001, vh / vv, 0)
    
    # Agua
    agua = np.zeros_like(vh, dtype=np.uint8)
    agua_mask = (vh < COEF_AGUA["umbral"]) & (vv < COEF_AGUA["umbral"])
    agua[agua_mask] = 255
    print(f"Agua: {np.sum(agua_mask)} ({100*np.sum(agua_mask)/agua_mask.size:.1f}%)")

    # Arrozales
    arrozal = np.zeros_like(vh, dtype=np.uint8)
    arrozal_mask = (
        (vh >= COEF_ARROZAL["vh_min"]) & (vh < COEF_ARROZAL["vh_max"]) &
        (vv < COEF_ARROZAL["vv_max"]) &
        (ratio > COEF_ARROZAL["ratio_min"])
    )
    arrozal[arrozal_mask] = 255
    print(f"Arrozales: {np.sum(arrozal_mask)} ({100*np.sum(arrozal_mask)/arrozal_mask.size:.1f}%)")

    # Zona quemada (valores absolutos muy bajos + ratio bajo)
    quemado = np.zeros_like(vh, dtype=np.uint8)
    tierra_seca_mask = ~(agua_mask | arrozal_mask)
    quemado_mask = (
        (vh < COEF_QUEMADO["vh_max"]) &
        (vv < COEF_QUEMADO["vv_max"]) &
        (ratio < COEF_QUEMADO["ratio_max"]) &
        tierra_seca_mask
    )
    quemado[quemado_mask] = 255
    print(f"Zona quemada: {np.sum(quemado_mask)} ({100*np.sum(quemado_mask)/quemado_mask.size:.1f}%)")

    # Vegetación (excluyendo quemado)
    vegetacion = np.zeros_like(vh, dtype=np.uint8)
    zona_no_quemada = tierra_seca_mask & ~quemado_mask
    veg_mask = (ratio >= COEF_VEGETACION["ratio_min"]) & zona_no_quemada
    vegetacion[veg_mask] = 255
    print(f"Vegetación: {np.sum(veg_mask)} ({100*np.sum(veg_mask)/veg_mask.size:.1f}%)")

    # Urbano
    urbano = np.zeros_like(vh, dtype=np.uint8)
    zona_no_clasificada = zona_no_quemada & ~veg_mask
    urbano_mask = (vv >= COEF_URBANO["vv_min"]) & zona_no_clasificada
    urbano[urbano_mask] = 255
    print(f"Urbano: {np.sum(urbano_mask)} ({100*np.sum(urbano_mask)/urbano_mask.size:.1f}%)")

    # Tierra
    tierra = np.zeros_like(vh, dtype=np.uint8)
    tierra_mask = ~(agua_mask | arrozal_mask | quemado_mask | veg_mask | urbano_mask)
    tierra[tierra_mask] = 255
    print(f"Tierra: {np.sum(tierra_mask)} ({100*np.sum(tierra_mask)/tierra_mask.size:.1f}%)")

    return agua, arrozal, vegetacion, urbano, tierra, quemado
    print("\n--- Estadísticas de datos SAR ---")
    print(f"VH - Min: {np.nanmin(vh):.2f}, Max: {np.nanmax(vh):.2f}, Media: {np.nanmean(vh):.2f}")
    print(f"VV - Min: {np.nanmin(vv):.2f}, Max: {np.nanmax(vv):.2f}, Media: {np.nanmean(vv):.2f}")

    diferencia = vv - vh
    print(f"VV-VH - Min: {np.nanmin(diferencia):.2f}, Max: {np.nanmax(diferencia):.2f}, Media: {np.nanmean(diferencia):.2f}")

    # Agua
    agua = np.zeros_like(vh, dtype=np.uint8)
    agua_mask = (vh < COEF_AGUA["umbral"]) & (vv < COEF_AGUA["umbral"])
    agua[agua_mask] = 255
    print(f"\nAgua: {np.sum(agua_mask)} ({100*np.sum(agua_mask)/agua_mask.size:.1f}%)")

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

    # Zona quemada (valores absolutos muy bajos + ratio bajo)
    quemado = np.zeros_like(vh, dtype=np.uint8)
    tierra_seca_mask = ~(agua_mask | arrozal_mask)
    quemado_mask = (
        (vh < COEF_QUEMADO["vh_max"]) &
        (vv < COEF_QUEMADO["vv_max"]) &
        (ratio < COEF_QUEMADO["ratio_max"]) &
        tierra_seca_mask
    )
    quemado[quemado_mask] = 255
    print(f"Zona quemada: {np.sum(quemado_mask)} ({100*np.sum(quemado_mask)/quemado_mask.size:.1f}%)")

    # Vegetación (excluyendo quemado)
    vegetacion = np.zeros_like(vh, dtype=np.uint8)
    zona_no_quemada = tierra_seca_mask & ~quemado_mask
    veg_mask = (ratio >= COEF_VEGETACION["ratio_min"]) & zona_no_quemada
    vegetacion[veg_mask] = 255
    print(f"Vegetación: {np.sum(veg_mask)} ({100*np.sum(veg_mask)/veg_mask.size:.1f}%)")

    # Urbano
    urbano = np.zeros_like(vh, dtype=np.uint8)
    zona_no_clasificada = zona_no_quemada & ~veg_mask
    urbano_mask = (vv >= COEF_URBANO["vv_min"]) & zona_no_clasificada
    urbano[urbano_mask] = 255
    print(f"Urbano: {np.sum(urbano_mask)} ({100*np.sum(urbano_mask)/urbano_mask.size:.1f}%)")

    # Tierra
    tierra = np.zeros_like(vh, dtype=np.uint8)
    tierra_mask = ~(agua_mask | arrozal_mask | quemado_mask | veg_mask | urbano_mask)
    tierra[tierra_mask] = 255
    print(f"Tierra: {np.sum(tierra_mask)} ({100*np.sum(tierra_mask)/tierra_mask.size:.1f}%)")

    return agua, arrozal, vegetacion, urbano, tierra, quemado




if __name__ == '__main__':
    main()