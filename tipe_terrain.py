import numpy as np
import rasterio
from PIL import Image
import os
from dotenv import load_dotenv

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
    "ratio_min": 0.5
}

# Urbano/rocoso
COEF_URBANO = {
    "vv_min": 2
}
def main():
    # Configuración
    load_dotenv()
    ruta_geotiff = os.getenv("RUTA_GEOTIFF1")
    
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