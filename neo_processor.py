# Este script solo es para uso local, no forma parte de la aplicación web.
# Descarga y procesa datos de NEOs desde la API de la NASA de manera mas eficiente a hacerlo desde php.

import requests
import pandas as pd
import json
import time

API_KEY = "DJ2obbZu2pZfGqJYABtQyRcEfSfmReXtyBtVJLI0"
BASE_URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"

# ---------------------------------------
# 1. DESCARGAR TODAS LAS PÁGINAS
# ---------------------------------------

def download_all_neos():
    all_neos = []
    page = 0

    while True:
        print(f"Descargando página: {page}")

        response = requests.get(BASE_URL, params={
            "api_key": API_KEY,
            "page": page
        })

        if response.status_code != 200:
            print("Error en petición:", response.text)
            break

        data = response.json()

        batch = data.get("near_earth_objects", [])
        all_neos.extend(batch)

        # Si no hay página siguiente, romper
        if "next" not in data.get("links", {}):
            break

        page += 1
        time.sleep(0.2)  # evitar límite NASA

    print(f"Total NEOs descargados: {len(all_neos)}\n")
    return all_neos

# ---------------------------------------
# 2. PROCESAR DISTANCIA MÍNIMA A LA TIERRA
# ---------------------------------------

def get_min_earth_distance(neo):
    min_dist = None

    for approach in neo.get("close_approach_data", []):
        if approach.get("orbiting_body") != "Earth":
            continue

        dist_km = float(approach["miss_distance"]["kilometers"])

        if min_dist is None or dist_km < min_dist:
            min_dist = dist_km

    return min_dist

# ---------------------------------------
# 3. PROCESAR DATASET Y ENCONTRAR EL MEJOR
# ---------------------------------------

def process_neos(all_neos):

    # Crear DataFrame con información clave
    rows = []
    for neo in all_neos:
        min_dist = get_min_earth_distance(neo)
        if min_dist is None:
            continue

        diameter = neo["estimated_diameter"]["meters"]["estimated_diameter_max"]

        rows.append({
            "neo_id": neo["id"],
            "name": neo["name"],
            "absolute_magnitude": neo["absolute_magnitude_h"],
            "diameter_m": diameter,
            "min_distance_km": min_dist,
            "full": neo
        })

    df = pd.DataFrame(rows)

    # Ordenar por distancia mínima
    df = df.sort_values(by="min_distance_km")

    # Tomar el 10% más cercano
    top_10_percent = int(len(df) * 0.10)
    df_top = df.head(top_10_percent)

    # Elegir el más grande del top 10%
    best_row = df_top.loc[df_top["diameter_m"].idxmax()]

    return df, best_row


# ---------------------------------------
# 4. EXPORTAR RESULTADOS
# ---------------------------------------

def export_results(df, best_row):
    df.to_csv("all_neos.csv", index=False)

    with open("best_candidate.json", "w") as f:
        json.dump(best_row["full"], f, indent=4)

    print("\nArchivos generados:")
    print(" - all_neos.csv (todos los asteroides)")
    print(" - best_candidate.json (mejor candidato)")


# ---------------------------------------
# MAIN
# ---------------------------------------

if __name__ == "__main__":
    print("=== DESCARGANDO BASE DE DATOS NEO ===")
    all_neos = download_all_neos()

    print("=== PROCESANDO Y BUSCANDO MEJOR CANDIDATO ===")
    df, best = process_neos(all_neos)

    print("\n=== ASTEROIDE IDEAL ENCONTRADO ===")
    print("Nombre:", best["name"])
    print("Distancia mínima (km):", best["min_distance_km"])
    print("Diámetro máximo (m):", best["diameter_m"])

    export_results(df, best)