import numpy as np
import rasterio
from PIL import Image
import os
from dotenv import load_dotenv

def leer_geotiff(ruta_archivo):
    """Lee GeoTIFF con bandas VH, VV y altura"""
    with rasterio.open(ruta_archivo) as src:
        vh = src.read(1)
        vv = src.read(2)
        altura = src.read(3)
        return vh, vv, altura

def clasificar_terreno(vh, vv):
    """Clasifica terreno en capas separadas - para datos SAR en dB"""
    # Mostrar estadísticas de los datos
    print("\n--- Estadísticas de datos SAR (dB) ---")
    print(f"VH - Min: {np.nanmin(vh):.2f}, Max: {np.nanmax(vh):.2f}, Media: {np.nanmean(vh):.2f}")
    print(f"VV - Min: {np.nanmin(vv):.2f}, Max: {np.nanmax(vv):.2f}, Media: {np.nanmean(vv):.2f}")
    
    # Diferencia VV-VH útil para clasificación
    diferencia = vv - vh
    print(f"VV-VH - Min: {np.nanmin(diferencia):.2f}, Max: {np.nanmax(diferencia):.2f}, Media: {np.nanmean(diferencia):.2f}")
    
    # Capa 1: Agua (valores muy bajos en dB, típicamente < -20 dB)
    # Agua suele tener backscatter muy bajo (valores muy negativos)
    umbral_agua = -100  # Ajustar según tus datos
    
    agua = np.zeros_like(vh, dtype=np.uint8)
    agua_mask = (vh < umbral_agua) & (vv < umbral_agua)
    agua[agua_mask] = 255
    
    print(f"Píxeles clasificados como agua: {np.sum(agua_mask)} de {agua_mask.size} ({100*np.sum(agua_mask)/agua_mask.size:.1f}%)")
    
    # Capa 2: Vegetación (diferencia VV-VH pequeña, VH relativamente alto)
    # Vegetación: diferencia pequeña entre VV y VH
    vegetacion = np.zeros_like(vh, dtype=np.uint8)
    # Normalizar diferencia a 0-255 (menos diferencia = más vegetación)
    dif_norm = np.clip((diferencia - np.nanmin(diferencia)) / (np.nanmax(diferencia) - np.nanmin(diferencia)), 0, 1)
    veg_values = ((1 - dif_norm) * 255).astype(np.uint8)  # Invertir: menos diferencia = más vegetación
    vegetacion[~agua_mask] = veg_values[~agua_mask]
    

    # Capa 3: Urbano/rocoso (VV alto, valores menos negativos)
    urbano = np.zeros_like(vh, dtype=np.uint8)
    # Normalizar VV a 0-255 (valores menos negativos = más urbano)
    vv_norm = np.clip((vv - np.nanmin(vv)) / (np.nanmax(vv) - np.nanmin(vv)), 0, 1)
    urbano_values = (vv_norm * 255).astype(np.uint8)
    urbano[~agua_mask] = urbano_values[~agua_mask]
    
    return agua, vegetacion, urbano

def normalizar_altura(altura, bits=16):
    """Normaliza altura a rango 0-65535 (16-bit) o 0-255 (8-bit)"""
    altura_norm = (altura - np.nanmin(altura)) / (np.nanmax(altura) - np.nanmin(altura))
    if bits == 16:
        altura_norm = (altura_norm * 65535).astype(np.uint16)
    else:
        altura_norm = (altura_norm * 255).astype(np.uint8)
    return altura_norm

def main():
    # Configuración
    load_dotenv()
    ruta_geotiff = os.getenv("RUTA_GEOTIFF")
    
    if not ruta_geotiff or not os.path.exists(ruta_geotiff):
        print("Error: RUTA_GEOTIFF no configurada o archivo no existe")
        return
    
    # Leer datos
    print(f"Leyendo {ruta_geotiff}...")
    vh, vv, altura = leer_geotiff(ruta_geotiff)
    
    # Clasificar terreno en capas
    print("Clasificando terreno en capas...")
    agua, vegetacion, urbano = clasificar_terreno(vh, vv)
    
    # Normalizar altura
    print("Procesando altura...")
    altura_16bit = normalizar_altura(altura, bits=16)
    
    # Guardar PNGs
    print("Guardando imágenes...")
    Image.fromarray(altura_16bit, mode='I;16').save('layer_height.png')
    Image.fromarray(agua, mode='L').save('layer_water.png')
    Image.fromarray(vegetacion, mode='L').save('layer_vegetation.png')
    Image.fromarray(urbano, mode='L').save('layer_urban.png')
    
    print("\n¡Completado!")
    print("\nCapas generadas:")
    print("- layer_height.png: Altura del terreno (16-bit)")
    print("- layer_water.png: Cuerpos de agua (255=agua, 0=tierra)")
    print("- layer_vegetation.png: Densidad de vegetación (0-255)")
    print("- layer_urban.png: Áreas urbanas/rocosas (0-255)")
    print("\nEn WorldPainter:")
    print("1. Importa layer_height.png como Height Map")
    print("2. Usa layer_water.png para crear lagos/ríos")
    print("3. Usa layer_vegetation.png para distribuir árboles/plantas")
    print("4. Usa layer_urban.png para zonas rocosas/construcciones")

if __name__ == '__main__':
    main()