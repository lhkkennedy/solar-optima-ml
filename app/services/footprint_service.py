import json
import time
from typing import Optional, Dict, Any
import requests
from dataclasses import dataclass

@dataclass
class Footprint:
    geojson: Dict[str, Any]
    provider: str = "osm"


class FootprintService:
    def __init__(self, overpass_url: str = "https://overpass-api.de/api/interpreter", timeout_s: int = 15):
        self.overpass_url = overpass_url
        self.timeout_s = timeout_s

    def get_building_polygon(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Optional[Footprint]:
        # Overpass bbox: south,west,north,east
        bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
        q = f"[out:json][timeout:15];way[building]({bbox});out geom;"
        try:
            r = requests.post(self.overpass_url, data={"data": q}, timeout=self.timeout_s, headers={"User-Agent": "solaroptima-ml/1.0"})
            r.raise_for_status()
            data = r.json()
            # Pick the largest building by number of nodes as a heuristic
            elements = data.get("elements", [])
            if not elements:
                return None
            elements.sort(key=lambda e: len(e.get("geometry", [])), reverse=True)
            best = elements[0]
            coords = [(pt["lon"], pt["lat"]) for pt in best.get("geometry", [])]
            if len(coords) < 3:
                return None
            geojson = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]}, "properties": {"source": "osm", "id": best.get("id")}}
            return Footprint(geojson=geojson)
        except Exception:
            return None