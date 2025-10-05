#!/usr/bin/env python3
"""
download_and_run_gpt_asf.py
Descarga productos Sentinel-1 GRD desde Alaska Satellite Facility (ASF)
usando tu cuenta y ejecuta SNAP GPT para procesarlos.
"""

import os
from pathlib import Path
import subprocess
from tqdm import tqdm
import asf_search as asf

# ========= CONFIGURACIÓN =========

# Carga credenciales ASF desde variables de entorno
ASF_USER = os.getenv("ASF_USER")
ASF_PASS = os.getenv("ASF_PASS")

if not ASF_USER or not ASF_PASS:
    raise RuntimeError("❌ No se encontraron las variables ASF_USER / ASF_PASS. Añádelas en tu ~/.zshrc y ejecuta `source ~/.zshrc`.")

# Fechas del incendio de Puebla de Sanabria (agosto 2025)
# Ventanas: antes y después
DATE_WINDOWS = {
    "pre":  ("2025-08-01", "2025-08-13"),
    "post": ("2025-08-14", "2025-08-25"),
}

# AOI: bounding box de Puebla de Sanabria (~25 km)
BBOX = (-6.85, 41.90, -6.42, 42.20)  # minlon, minlat, maxlon, maxlat

# Rutas locales
BASE_DIR = Path("./asf_sanabria")
DOWNLOAD_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "geotiffs"
GRAPH_XML = Path("./GptGraph_S1_GRD_to_GeoTIFF.xml")
GPT_CMD = "gpt"

# ========= FUNCIONES =========

def bbox_to_wkt(bbox):
    minlon, minlat, maxlon, maxlat = bbox
    return f"POLYGON(({minlon} {minlat}, {minlon} {maxlat}, {maxlon} {maxlat}, {maxlon} {minlat}, {minlon} {minlat}))"

def ensure_dirs():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def search_asf_products(wkt, start, end):
    """Busca productos Sentinel-1 GRD (IW) en ASF."""
    results = asf.search(
        platform="Sentinel-1",
        processingLevel="GRD",
        beamMode="IW",
        intersectsWith=wkt,
        start=start,
        end=end,
    )
    return results

def download_products(products, outdir):
    """Descarga los productos a la carpeta indicada."""
    for prod in tqdm(products, desc="Descargando desde ASF"):
        local_path = outdir / f"{prod.properties['fileName']}.zip"
        if local_path.exists():
            print(f"✅ Ya existe {local_path.name}, omitiendo descarga.")
            continue
        prod.download(path=str(outdir), session=asf.ASFSession().auth_with_creds(ASF_USER, ASF_PASS))

def run_gpt(graph_xml, input_safe, output_tif):
    """Ejecuta SNAP GPT sobre un producto .SAFE/.zip"""
    # Sustituimos las variables en el XML temporalmente
    xml_text = graph_xml.read_text(encoding='utf-8')
    xml_text = xml_text.replace("${INPUT_SAFE}", str(input_safe))
    xml_text = xml_text.replace("${OUTPUT_GEOTIFF}", str(output_tif))
    tmp_xml = output_tif.with_suffix(".xml")
    tmp_xml.write_text(xml_text, encoding='utf-8')

    cmd = [GPT_CMD, str(tmp_xml)]
    print(f"[GPT] Procesando {input_safe.name}...")
    subprocess.run(cmd, check=True)

def process_all():
    ensure_dirs()
    wkt = bbox_to_wkt(BBOX)
    for label, (start, end) in DATE_WINDOWS.items():
        print(f"\n=== Ventana {label}: {start} → {end} ===")
        results = search_asf_products(wkt, start, end)
        if not results:
            print("⚠️ No se encontraron productos ASF.")
            continue
        folder = DOWNLOAD_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        download_products(results, folder)

        out_subdir = OUTPUT_DIR / label
        out_subdir.mkdir(parents=True, exist_ok=True)
        for prod in results:
            safe_zip = folder / f"{prod.properties['fileName']}.zip"
            out_tif = out_subdir / f"{prod.properties['fileName']}_proc.tif"
            run_gpt(GRAPH_XML, safe_zip, out_tif)

if __name__ == "__main__":
    process_all()
