import numpy as np
import rasterio
from amulet import Block
from amulet.api.level import World
from amulet.api.chunk import Chunk
import os

def leer_geotiff(ruta_archivo):
    """
    Lee un archivo GeoTIFF con bandas VH, VV y altura
    Retorna los arrays de cada banda y metadatos
    """
    with rasterio.open(ruta_archivo) as src:
        # Asumiendo que las bandas están en orden: VH, VV, Altura
        vh = src.read(1)
        vv = src.read(2)
        altura = src.read(3)
        
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height
        }
        
        return vh, vv, altura, metadata

def clasificar_terreno(vh, vv, altura):
    """
    Clasifica el terreno basándose en valores SAR y altura
    Retorna una matriz con códigos de tipo de terreno
    """
    terreno = np.zeros_like(vh, dtype=int)
    
    # Normalizar valores SAR (ajustar según tus datos)
    vh_norm = (vh - np.nanmin(vh)) / (np.nanmax(vh) - np.nanmin(vh))
    vv_norm = (vv - np.nanmin(vv)) / (np.nanmax(vv) - np.nanmin(vv))
    
    # Ratio VH/VV útil para discriminar superficies
    ratio = np.where(vv != 0, vh / vv, 0)
    
    # Clasificación simple:
    # 0 - Agua (baja retrodispersión en ambas polarizaciones)
    # 1 - Vegetación densa (alta VH, ratio alto)
    # 2 - Tierra/suelo (valores medios)
    # 3 - Urbano/rocoso (alta VV)
    # 4 - Nieve/hielo (muy alta retrodispersión)
    
    # Agua: backscatter muy bajo
    agua_mask = (vh_norm < 0.2) & (vv_norm < 0.2)
    terreno[agua_mask] = 0
    
    # Vegetación: ratio VH/VV alto
    veg_mask = (ratio > 0.5) & (~agua_mask)
    terreno[veg_mask] = 1
    
    # Urbano/rocoso: VV alto
    urbano_mask = (vv_norm > 0.7) & (~agua_mask) & (~veg_mask)
    terreno[urbano_mask] = 3
    
    # Nieve/hielo: valores muy altos en ambas
    nieve_mask = (vh_norm > 0.8) & (vv_norm > 0.8)
    terreno[nieve_mask] = 4
    
    # El resto es tierra/suelo
    terreno[(terreno == 0) & (~agua_mask) & (~veg_mask) & (~urbano_mask) & (~nieve_mask)] = 2
    
    return terreno

def tipo_terreno_a_bloque(tipo):
    """
    Mapea el tipo de terreno a bloques de Minecraft
    """
    bloques = {
        0: Block('minecraft', 'water'),           # Agua
        1: Block('minecraft', 'grass_block'),     # Vegetación
        2: Block('minecraft', 'dirt'),            # Tierra
        3: Block('minecraft', 'stone'),           # Urbano/rocoso
        4: Block('minecraft', 'snow_block')       # Nieve
    }
    return bloques.get(tipo, Block('minecraft', 'stone'))

def crear_mundo_minecraft(vh, vv, altura, nombre_mundo='mundo_sar', escala_altura=1.0):
    """
    Crea un mundo de Minecraft a partir de datos GeoTIFF
    """
    # Clasificar terreno
    print("Clasificando terreno...")
    terreno = clasificar_terreno(vh, vv, altura)
    
    # Normalizar altura (ajustar al rango de Minecraft -64 a 319)
    altura_norm = ((altura - np.nanmin(altura)) / 
                   (np.nanmax(altura) - np.nanmin(altura)) * 100 * escala_altura + 64)
    altura_norm = np.clip(altura_norm, -64, 319).astype(int)
    
    # Crear mundo
    print("Creando mundo de Minecraft...")
    world = World.create_world(nombre_mundo, ('java', (1, 18, 0)))
    
    # Submuestrear si el GeoTIFF es muy grande
    filas, cols = vh.shape
    max_dim = 512  # Máximo tamaño recomendado
    
    if filas > max_dim or cols > max_dim:
        print(f"Submuestreando de {filas}x{cols} a máximo {max_dim}x{max_dim}...")
        factor = max(filas // max_dim, cols // max_dim) + 1
        terreno = terreno[::factor, ::factor]
        altura_norm = altura_norm[::factor, ::factor]
        filas, cols = terreno.shape
    
    print(f"Generando mundo de {filas}x{cols} bloques...")
    
    # Generar terreno
    for x in range(cols):
        for z in range(filas):
            if np.isnan(altura[z, x]):
                continue
                
            h = altura_norm[z, x]
            tipo = terreno[z, x]
            bloque_superficie = tipo_terreno_a_bloque(tipo)
            
            # Colocar bedrock en el fondo
            world.set_version_block(x, -64, z, 'minecraft:bedrock', 
                                   ('java', (1, 18, 0)))
            
            # Llenar con stone hasta la superficie
            for y in range(-63, h):
                world.set_version_block(x, y, z, 'minecraft:stone', 
                                       ('java', (1, 18, 0)))
            
            # Colocar bloque de superficie
            world.set_version_block(x, h, z, bloque_superficie, 
                                   ('java', (1, 18, 0)))
            
            # Si es vegetación, añadir capas de tierra debajo
            if tipo == 1 and h > -63:
                for y in range(max(-63, h-3), h):
                    world.set_version_block(x, y, z, 'minecraft:dirt', 
                                           ('java', (1, 18, 0)))
        
        if x % 50 == 0:
            print(f"Progreso: {x}/{cols} columnas")
    
    # Guardar mundo
    print("Guardando mundo...")
    world.save()
    world.close()
    print(f"Mundo guardado en: {nombre_mundo}")

def main():
    """
    Función principal
    """
    # Configuración
    ruta_geotiff = 'datos_sar.tif'  # Cambia esto por tu archivo
    nombre_mundo = 'mundo_sar_minecraft'
    escala_altura = 1.0  # Ajusta para exagerar o reducir el relieve
    
    # Verificar que existe el archivo
    if not os.path.exists(ruta_geotiff):
        print(f"Error: No se encuentra el archivo {ruta_geotiff}")
        return
    
    # Leer GeoTIFF
    print(f"Leyendo archivo {ruta_geotiff}...")
    vh, vv, altura, metadata = leer_geotiff(ruta_geotiff)
    
    print(f"Dimensiones: {metadata['width']}x{metadata['height']}")
    print(f"Rango VH: {np.nanmin(vh):.2f} a {np.nanmax(vh):.2f}")
    print(f"Rango VV: {np.nanmin(vv):.2f} a {np.nanmax(vv):.2f}")
    print(f"Rango altura: {np.nanmin(altura):.2f} a {np.nanmax(altura):.2f}")
    
    # Crear mundo
    crear_mundo_minecraft(vh, vv, altura, nombre_mundo, escala_altura)
    
    print("\n¡Proceso completado!")
    print(f"Abre el mundo '{nombre_mundo}' en Minecraft Java Edition")

if __name__ == '__main__':
    main()