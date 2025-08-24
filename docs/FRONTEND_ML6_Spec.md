# Frontend Spec: ML‑6 Roof 3D Parametrization

## Goal
Let users pick a location (lat/lon) and optionally upload imagery to run the `/model3d` pipeline and visualize planes/edges, plus a quick `/pitch` summary.

## Tech (suggested)
- Next.js (React, TS), React Query (data), Zustand (UI state)
- Map: Mapbox GL JS or Leaflet; overlays via deck.gl
- 3D preview: three.js or react-three-fiber (glTF)
- Styling: Tailwind or CSS Modules

## Routes
- `/` Analyzer
- `/share?lat=..&lon=..&bbox=..` (optional deep-link)

## UI
- **Map panel**
  - Search (geocode), “Use map center”
  - Draggable square bbox overlay (40–120 m; default 60 m)
- **Controls panel**
  - Lat/Lon (read‑only), BBox slider
  - Image upload (PNG/JPG ≤10 MB) or URL
  - Toggle: Return 3D mesh
  - Buttons: Run Model 3D, Quick Pitch
- **Results panel**
  - Summary: area, pitches/orientations, max height, confidence
  - Toggle overlays: Planes, Ridges
  - Downloads: GeoJSON, glTF (when present)
  - Request ID; elapsed time

## API Contracts
### POST /model3d
Request:
```json
{
  "coordinates": {"latitude": 51.5074, "longitude": -0.1278},
  "bbox_m": 60,
  "image_base64": "<optional>",
  "image_url": "<optional>",
  "provider_hint": "client|url",
  "return_mesh": true
}
```
Response (subset):
```json
{
  "planes":[{"id":"p1","normal":[0.1,-0.9,0.4],"pitch_deg":23.4,"aspect_deg":178.0,
    "polygon":[[lon,lat,h]...],"area_m2":45.2}],
  "edges":[{"a":[lon,lat,h],"b":[lon,lat,h]}],
  "summary":{"area_m2":140.3,"max_height_m":11.8,"roof_type":"gabled"},
  "confidence":0.89,
  "artifacts":{"geojson_url":"...","gltf_url":"..."}
}
```

### POST /pitch (fast path)
Returns lightweight summary (pitch_degrees, area_m2, orientation, roof_type, confidence).

## Overlay Behavior
- Planes → GeoJSON Polygon (fill by pitch w/ legend)
- Edges → LineString (label ridges)
- Click plane → tooltip: pitch°, area m², aspect°
- Mesh view loads `artifacts.gltf_url` in three.js canvas

## State & Errors
- React Query for calls; show spinner with request_id
- Validate: UK bounds, bbox_m ∈ [40,120], image type/size; client‑side downscale ≤2048 px
- Error messages: Out of bounds, DSM unavailable, Rate‑limited, Upload too large

## Performance Budgets
- P95 ≤ 5 s for first result with cache
- Map render ≤ 60 ms/frame; mesh < 10 MB

## Accessibility
- Keyboard operable controls; visible focus styles
- Color‑blind‑safe palette for pitch legend; ARIA labels for controls

## Security/CORS
- HTTPS; CORS allow frontend origin; no secrets in browser (future API key remains server‑side)

## Telemetry
- Log request_id, lat/lon (coarsened), bbox_m, hasImage, duration, outcome
- Basic UX metrics: time to first polygon, mesh render FPS

## Env Config
- `NEXT_PUBLIC_API_BASE_URL`
- `NEXT_PUBLIC_MAP_TOKEN` (Mapbox)
- `MAX_UPLOAD_MB=10`, `DEFAULT_BBOX_M=60`

## Definition of Done
- User selects location, adjusts bbox, uploads or links image, clicks Run
- Sees polygons/edges on map, summary, confidence; downloads GeoJSON; optional glTF preview
- “Quick Pitch” works via `/pitch`; CORS set; errors handled; responsive and basic a11y

## Nice‑to‑have (v2)
- “Snap to building” (footprint toggle)
- Shareable link (query params)
- Batch CSV upload (lat,lon)
- Export PDF snapshot of summary + map screenshot