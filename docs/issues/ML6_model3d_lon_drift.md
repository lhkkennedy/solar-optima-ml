## ML-6 bug: /model3d returns geometry far from drawn bbox (longitude drift)

### Summary
When posting to `/model3d` with a bbox around lon ≈ -1.796, lat ≈ 51.559 (Oxford area), the API response contains `planes[].polygon` and `bbox.epsg4326` centered around lon ≈ -1.900 (≈ 7 km west). Latitude is correct; longitude is shifted by ~0.10°.

### Impact
- Visual overlays do not align with the user-selected area in the demo.
- Any downstream artifact storage/links will reference incorrect geographies.

### Reproduction (dev)
1) Open `web/preview/index.html` via a local server (see README).
2) Enter service URL: your dev deployment.
3) Draw a small bbox over a known site near lon -1.796, lat 51.559.
4) Click “Capture satellite under bbox” and “Send to /model3d”.
5) Observe response vs drawn bbox.

Example drawn bbox (south, west, north, east):
```
[51.558803428775235, -1.7969886398122517, 51.559128601644574, -1.7963532619301041]
```

Example response (excerpt):
```json
{
  "planes": [
    {
      "polygon": [
        [-1.9007018082190188, 51.558767817011706, 0.44295],
        [-1.9000274183889738, 51.558767817011706, 0.77719],
        [-1.9000274183889738, 51.55896445459417, 0.78078],
        [-1.9007018082190188, 51.55896445459417, 0.44655]
      ]
    }
  ],
  "bbox": {
    "epsg4326": [-1.9007018082190188, 51.558767817011706, -1.9000274183889738, 51.5591642134081]
  }
}
```

### Expected
- Response `bbox.epsg4326` and `planes[].polygon` lie within the user’s input bbox (same longitudes to within a few meters).

### Actual
- Longitudes are ~0.10° west of the input (~7 km); latitudes are correct.

### Likely root causes to investigate
- Axis/order mismatches between lon/lat vs lat/lon in helpers around projection transforms.
- `EPSG:27700` (OSGB) uses (x=easting, y=northing). Verify we consistently treat tuples as `(min_x, min_y, max_x, max_y)`.
- Explicitly enforce `always_xy=True` for any pyproj/transformer usage (if present).
- Confirm `locate_and_fetch(lat, lon, bbox_m)` parameter order matches all downstream computations.
- Validate `bbox_projected_around`, `bbox_to_wgs84`, and any `DsmIndexer` tile bbox math.

### Diagnostics to add (temporary logging)
- Log request center `(lat, lon)` and computed `bbox_m`.
- Log projected center `(x, y)` in EPSG:27700 and projected bbox `(minx, miny, maxx, maxy)`.
- Log round-trip WGS84 bbox after `bbox_to_wgs84`.
- If round-trip lon differs by > 1e-4°, flag a warning.

### Suggested unit tests
- Round-trip tests for known point(s):
  - Input: `(lat=51.5589, lon=-1.7969, bbox_m=60)` → project → bbox → inverse to WGS84.
  - Assert output bbox min/max lon/lat within 5 meters equivalent of input-derived bbox.
- Plane overlay sanity: ensure all `planes[].polygon` lon/lat lie within returned `bbox.epsg4326`.

### Acceptance criteria
- For 3 sites (urban/suburban/rural), response bbox and polygons lie within the drawn bbox with ≤ 5 m positional error.
- All new tests pass locally and in CI.
- Temporary debug logs removed or downgraded to trace.

### References (code hotspots)
- `app/services/dsm_service.py`: bbox construction, WCS requests, conversions
- `app/services/dsm_index.py`: tile references and bbox math
- `app/main.py`: `/model3d` request parsing and call to `locate_and_fetch`

### Notes
- The demo visualizer overlays planes on the Leaflet map; current misalignment strongly suggests a systematic longitude transform issue rather than rendering.
