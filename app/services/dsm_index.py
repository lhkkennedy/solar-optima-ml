import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

@dataclass
class TileRef:
    id: str
    url_dsm: str
    url_dtm: Optional[str]
    resolution_m: float
    bbox_27700: Optional[Tuple[float, float, float, float]] = None


class DsmIndexer:
    """Tile indexer for EA LIDAR DSM/DTM.
    - If LIDAR_INDEX_JSON is set, loads index records from JSON (list of {bbox_27700, res_m, dsm_url, dtm_url}).
    - Else falls back to a simple 1km grid with templated URLs (scaffold).
    """

    def __init__(self, base_url: str, resolution_m: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.resolution_m = resolution_m
        self._index = self._load_json_index(os.getenv("LIDAR_INDEX_JSON"))

    def tiles_for_bbox(self, bbox_27700: Tuple[float, float, float, float]) -> List[TileRef]:
        if self._index:
            minx, miny, maxx, maxy = bbox_27700
            out: List[TileRef] = []
            for rec in self._index:
                if abs(rec.get("res_m", self.resolution_m) - self.resolution_m) > 0.1:
                    continue
                bx, by, bX, bY = rec["bbox_27700"]
                # simple intersects check
                if not (bX < minx or bY < miny or bx > maxx or by > maxy):
                    out.append(TileRef(
                        id=rec.get("id", f"{int(bx)}_{int(by)}"),
                        url_dsm=rec["dsm_url"],
                        url_dtm=rec.get("dtm_url"),
                        resolution_m=rec.get("res_m", self.resolution_m),
                        bbox_27700=(bx, by, bX, bY),
                    ))
            return out
        return self._grid_tiles(bbox_27700)

    def _grid_tiles(self, bbox_27700: Tuple[float, float, float, float]) -> List[TileRef]:
        minx, miny, maxx, maxy = bbox_27700
        size = 1000.0
        x0 = math.floor(minx / size) * size
        y0 = math.floor(miny / size) * size
        tiles: List[TileRef] = []
        x = x0
        while x < maxx:
            y = y0
            while y < maxy:
                tile_id = f"{int(x)}_{int(y)}_{int(size)}"
                url_dsm = f"{self.base_url}/dsm/{int(self.resolution_m)}m/{tile_id}.tif"
                url_dtm = f"{self.base_url}/dtm/{int(self.resolution_m)}m/{tile_id}.tif"
                tiles.append(TileRef(id=tile_id, url_dsm=url_dsm, url_dtm=url_dtm, resolution_m=self.resolution_m,
                                     bbox_27700=(x, y, x+size, y+size)))
                y += size
            x += size
        return tiles

    def _load_json_index(self, path: Optional[str]) -> Optional[List[dict]]:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # normalize
            out = []
            for rec in data:
                bx = tuple(rec.get("bbox_27700"))
                if len(bx) != 4:
                    continue
                out.append({
                    "id": rec.get("id"),
                    "bbox_27700": (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])),
                    "res_m": float(rec.get("res_m", self.resolution_m)),
                    "dsm_url": rec.get("dsm_url", ""),
                    "dtm_url": rec.get("dtm_url"),
                })
            return out
        except Exception:
            return None