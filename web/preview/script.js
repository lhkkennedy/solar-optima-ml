// Minimal preview UI for ML-6: draw bbox, send image + bbox to /model3d

const $ = (sel) => document.querySelector(sel);
const serviceUrlInput = $('#serviceUrl');
const saveUrlBtn = $('#saveUrl');
const bboxOut = $('#bboxOut');
const clearBboxBtn = $('#clearBbox');
const zoomUkBtn = $('#zoomUK');
const fileInput = $('#fileInput');
const clearImageBtn = $('#clearImage');
const imgInfo = $('#imgInfo');
const sendModel3dBtn = $('#sendModel3d');
const sendInferBtn = $('#sendInfer');
const responseEl = $('#response');
const basemapSelect = $('#basemap');
const captureSatBtn = $('#captureSat');
const artifactLinks = $('#artifactLinks');
const gltfViewer = document.getElementById('gltfViewer');
const overlayBtn = document.getElementById('overlayGeoJson');
const overlayPlanesBtn = document.getElementById('overlayPlanes');
const fitOverlayBtn = document.getElementById('fitOverlay');
const clearOverlayBtn = document.getElementById('clearOverlay');
const autoOverlayChk = document.getElementById('autoOverlay');
const rendererModeSel = document.getElementById('rendererMode');
const strokeW = document.getElementById('strokeW');
const fillA = document.getElementById('fillA');
const strokeC = document.getElementById('strokeC');
const fillC = document.getElementById('fillC');
const showVertices = document.getElementById('showVertices');
const showCentroid = document.getElementById('showCentroid');
const jsonInput = document.getElementById('jsonInput');

let imageBase64 = null;
let bboxWgs84 = null; // [south, west, north, east]
let lastResponseJson = null;

// Persist service URL
const savedUrl = localStorage.getItem('service_url');
if (savedUrl) serviceUrlInput.value = savedUrl;
saveUrlBtn.addEventListener('click', (e) => {
  e.preventDefault();
  const v = (serviceUrlInput.value || '').trim();
  if (v) {
    localStorage.setItem('service_url', v);
    responseEl.textContent = 'Service URL saved';
  } else {
    responseEl.textContent = 'Please enter a Service URL';
  }
});
serviceUrlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    const v = (serviceUrlInput.value || '').trim();
    if (v) {
      localStorage.setItem('service_url', v);
      responseEl.textContent = 'Service URL saved';
    }
  }
});

// Map
const map = L.map('map', { preferCanvas: true }).setView([52.5, -1.7], 6); // UK view
// Custom pane to ensure our overlays appear above the drawn bbox
map.createPane('planesPane');
map.getPane('planesPane').style.zIndex = 650;
// Dedicated canvas renderer for planes pane (required when preferCanvas is true)
let planesRenderer = L.canvas({ pane: 'planesPane' });
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap contributors', crossOrigin: true
});
const esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 19, attribution: 'Tiles &copy; Esri', crossOrigin: true
});
esriSat.addTo(map);

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);
const drawControl = new L.Control.Draw({
  draw: {
    polygon: false,
    marker: false,
    circle: false,
    circlemarker: false,
    polyline: false,
    rectangle: {
      shapeOptions: { color: '#ef4444' }
    }
  },
  edit: { featureGroup: drawnItems }
});
map.addControl(drawControl);

map.on(L.Draw.Event.CREATED, (e) => {
  drawnItems.clearLayers();
  drawnItems.addLayer(e.layer);
  const b = e.layer.getBounds();
  // Leaflet returns [southWest(lat,lng), northEast(lat,lng)]
  bboxWgs84 = [b.getSouth(), b.getWest(), b.getNorth(), b.getEast()];
  bboxOut.textContent = JSON.stringify(bboxWgs84);
});

clearBboxBtn.addEventListener('click', () => {
  drawnItems.clearLayers();
  bboxWgs84 = null;
  bboxOut.textContent = 'No bbox yet';
});

zoomUkBtn.addEventListener('click', () => {
  map.setView([52.5, -1.7], 6);
});

basemapSelect.addEventListener('change', () => {
  if (basemapSelect.value === 'satellite') {
    map.removeLayer(osm);
    esriSat.addTo(map);
  } else {
    map.removeLayer(esriSat);
    osm.addTo(map);
  }
});

// Image handling
fileInput.addEventListener('change', async () => {
  const f = fileInput.files?.[0];
  if (!f) { imageBase64 = null; imgInfo.textContent = 'No image selected'; return; }
  const buf = await f.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  imageBase64 = btoa(binary);
  imgInfo.textContent = `${f.name} (${Math.round(f.size/1024)} KB)`;
});

clearImageBtn.addEventListener('click', () => {
  fileInput.value = '';
  imageBase64 = null;
  imgInfo.textContent = 'No image selected';
});

function getServiceUrl() {
  let url = (serviceUrlInput?.value || '').trim();
  if (!url) {
    const stored = (localStorage.getItem('service_url') || '').trim();
    if (stored) {
      serviceUrlInput.value = stored;
      url = stored;
    }
  }
  if (!url) throw new Error('Please enter Service URL in the top bar and click "Use URL"');
  return url.replace(/\/$/, '');
}

async function callModel3d() {
  try {
    if (!bboxWgs84) throw new Error('Draw a bbox on the map first');
    if (!imageBase64) throw new Error('Select an image first');
    // API expects center coordinates and a bbox size in meters
    const south = bboxWgs84[0], west = bboxWgs84[1], north = bboxWgs84[2], east = bboxWgs84[3];
    const latC = (south + north) / 2;
    const lonC = (west + east) / 2;
    const metersPerDegLat = 111320; // approx
    const metersPerDegLon = 111320 * Math.cos(latC * Math.PI / 180);
    const heightM = Math.abs(north - south) * metersPerDegLat;
    const widthM = Math.abs(east - west) * Math.max(1e-6, metersPerDegLon);
    const bboxM = Math.ceil(Math.max(widthM, heightM));
    const body = {
      coordinates: { latitude: latC, longitude: lonC },
      bbox_m: Math.max(40, Math.min(120, bboxM)),
      image_base64: imageBase64,
      return_mesh: true
    };
    const res = await fetch(getServiceUrl() + '/model3d', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const text = await res.text();
    responseEl.textContent = text;
    try {
      const json = JSON.parse(text);
      lastResponseJson = json;
      renderArtifacts(json);
      await maybeVisualize(json);
    } catch {}
  } catch (err) {
    responseEl.textContent = 'Error: ' + (err?.message || String(err));
  }
}

async function callInfer() {
  try {
    if (!imageBase64) throw new Error('Select an image first');
    // Send multipart form with the original file if available; otherwise send base64 as a file
    const form = new FormData();
    const f = fileInput.files?.[0];
    if (f) {
      form.append('file', f, f.name);
    } else {
      const blob = new Blob([Uint8Array.from(atob(imageBase64), c => c.charCodeAt(0))], { type: 'image/png' });
      form.append('file', blob, 'image.png');
    }
    const res = await fetch(getServiceUrl() + '/infer', { method: 'POST', body: form });
    const text = await res.text();
    responseEl.textContent = text;
  } catch (err) {
    responseEl.textContent = 'Error: ' + (err?.message || String(err));
  }
}

function renderArtifacts(json) {
  artifactLinks.innerHTML = '';
  const urls = [];
  // Try common fields
  for (const key of Object.keys(json || {})) {
    const v = json[key];
    if (typeof v === 'string' && (v.endsWith('.gltf') || v.endsWith('.glb') || v.endsWith('.geojson'))) {
      urls.push({ key, url: v });
    }
    if (v && typeof v === 'object') {
      for (const [k2, v2] of Object.entries(v)) {
        if (typeof v2 === 'string' && (v2.endsWith('.gltf') || v2.endsWith('.glb') || v2.endsWith('.geojson'))) {
          urls.push({ key: `${key}.${k2}`, url: v2 });
        }
      }
    }
  }
  if (!urls.length) return;
  urls.forEach(({ key, url }) => {
    const a = document.createElement('a');
    a.href = url; a.textContent = key.split('.').pop() + ' ↗'; a.target = '_blank'; a.className = 'btn';
    artifactLinks.appendChild(a);
  });
}

async function maybeVisualize(json) {
  try {
    if (!json) return;
    const gltfUrl = findFirstUrl(json, ['gltf', 'glb']);
    if (gltfUrl && gltfViewer) {
      gltfViewer.src = gltfUrl;
    }
    const gjUrl = findFirstUrl(json, ['geojson']);
    if (autoOverlayChk?.checked) {
      if (gjUrl) {
        await overlayGeoJson(gjUrl);
      } else if (Array.isArray(json?.planes)) {
        overlayPlanes(json);
      }
    }
  } catch (e) {
    console.warn('Visualize error', e);
  }
}

function findFirstUrl(obj, exts) {
  if (!obj) return null;
  for (const [k, v] of Object.entries(obj)) {
    if (typeof v === 'string' && exts.some(ext => v.toLowerCase().endsWith('.' + ext))) return v;
    if (v && typeof v === 'object') {
      const r = findFirstUrl(v, exts);
      if (r) return r;
    }
  }
  return null;
}

let geoJsonLayer = null;
let planeLayer = null;
async function overlayGeoJson(url) {
  try {
    const res = await fetch(url);
    const gj = await res.json();
    clearOverlays();
    geoJsonLayer = L.geoJSON(gj, { style: { color: '#10b981', weight: 2 } }).addTo(map);
    try { map.fitBounds(geoJsonLayer.getBounds(), { padding: [20, 20] }); } catch {}
  } catch (e) {
    responseEl.textContent = 'GeoJSON overlay error: ' + (e?.message || e);
  }
}

function overlayPlanes(json) {
  try {
    clearOverlays();
    const group = L.layerGroup();
    const planes = json?.planes || [];
    console.log('overlayPlanes: count', planes.length);
    planes.forEach(p => {
      if (!Array.isArray(p.polygon)) return;
      const latlngs = p.polygon.map(([lon, lat]) => [Number(lat), Number(lon)]);
      console.log('plane latlngs', latlngs);
      const poly = L.polygon(latlngs, { pane: 'planesPane', renderer: planesRenderer, color: strokeC.value, fillColor: fillC.value, weight: Number(strokeW.value || 5), fillOpacity: Number(fillA.value || 0.25) });
      poly.bindTooltip(`pitch: ${Number(p.pitch_deg).toFixed(2)}°\narea: ${Number(p.area_m2).toFixed(1)} m²`);
      group.addLayer(poly);
      // centroid marker for debug visibility
      if (showCentroid.checked) {
        const cLat = latlngs.reduce((s, ll) => s + ll[0], 0) / latlngs.length;
        const cLon = latlngs.reduce((s, ll) => s + ll[1], 0) / latlngs.length;
        group.addLayer(L.circleMarker([cLat, cLon], { pane: 'planesPane', renderer: planesRenderer, radius: 6, color: '#ef4444', weight: 3, fillOpacity: 0.9 }));
      }
      if (showVertices.checked) {
        latlngs.forEach(v => group.addLayer(L.circleMarker(v, { pane: 'planesPane', renderer: planesRenderer, radius: 3, color: '#06b6d4', weight: 2, fillOpacity: 0.9 })));
      }
    });
    if (Array.isArray(json.edges)) {
      json.edges.forEach(e => {
        if (!e?.a || !e?.b) return;
        const a = [Number(e.a[1]), Number(e.a[0])];
        const b = [Number(e.b[1]), Number(e.b[0])];
        group.addLayer(L.polyline([a, b], { pane: 'planesPane', renderer: planesRenderer, color: '#f59e0b', weight: 4 }));
      });
    }
    // If no planes, draw bbox from response for debugging
    if (!planes.length && json?.bbox?.epsg4326) {
      const [minLon, minLat, maxLon, maxLat] = json.bbox.epsg4326;
      const rect = L.rectangle([[minLat, minLon], [maxLat, maxLon]], { pane: 'planesPane', renderer: planesRenderer, color: '#10b981', weight: 3, fillOpacity: 0.1 });
      group.addLayer(rect);
    }
    planeLayer = group.addTo(map);
    // Bring shapes to front if supported; LayerGroup itself doesn't have bringToFront
    try { group.eachLayer(function(l){ if (l && typeof l.bringToFront === 'function') { l.bringToFront(); }}); } catch (e) { console.warn('bringToFront skipped', e); }
    try {
      const bounds = group.getBounds();
      if (bounds.isValid()) map.fitBounds(bounds.pad(0.2));
    } catch {}
  } catch (e) {
    console.warn('overlayPlanes error', e);
  }
}

function clearOverlays() {
  if (geoJsonLayer) { map.removeLayer(geoJsonLayer); geoJsonLayer = null; }
  if (planeLayer) { map.removeLayer(planeLayer); planeLayer = null; }
}

overlayBtn?.addEventListener('click', async () => {
  const jsonText = responseEl.textContent || '';
  try { const json = JSON.parse(jsonText); const gj = findFirstUrl(json, ['geojson']); if (gj) await overlayGeoJson(gj); else responseEl.textContent = 'No GeoJSON URL found in response'; } catch { responseEl.textContent = 'Response is not JSON'; }
});

clearOverlayBtn?.addEventListener('click', () => { clearOverlays(); });

overlayPlanesBtn?.addEventListener('click', () => {
  if (lastResponseJson) {
    overlayPlanes(lastResponseJson);
  } else {
    const txt = responseEl.textContent || '';
    try { const json = JSON.parse(txt); lastResponseJson = json; overlayPlanes(json); } catch { responseEl.textContent = 'Response is not JSON'; }
  }
});

fitOverlayBtn?.addEventListener('click', () => {
  if (planeLayer) {
    try { const b = planeLayer.getBounds(); if (b.isValid()) map.fitBounds(b.pad(0.2)); } catch {}
  } else {
    responseEl.textContent = 'No overlay to fit';
  }
});

rendererModeSel?.addEventListener('change', () => {
  planesRenderer = rendererModeSel.value === 'svg' ? L.svg({ pane: 'planesPane' }) : L.canvas({ pane: 'planesPane' });
  if (lastResponseJson) overlayPlanes(lastResponseJson);
});

strokeW?.addEventListener('input', () => { if (lastResponseJson) overlayPlanes(lastResponseJson); });
fillA?.addEventListener('input', () => { if (lastResponseJson) overlayPlanes(lastResponseJson); });
strokeC?.addEventListener('input', () => { if (lastResponseJson) overlayPlanes(lastResponseJson); });
fillC?.addEventListener('input', () => { if (lastResponseJson) overlayPlanes(lastResponseJson); });
showVertices?.addEventListener('change', () => { if (lastResponseJson) overlayPlanes(lastResponseJson); });
showCentroid?.addEventListener('change', () => { if (lastResponseJson) overlayPlanes(lastResponseJson); });

document.getElementById('renderFromJson')?.addEventListener('click', () => {
  try { const json = JSON.parse(jsonInput.value); lastResponseJson = json; overlayPlanes(json); }
  catch (e) { responseEl.textContent = 'Invalid JSON'; }
});

sendModel3dBtn.addEventListener('click', callModel3d);
sendInferBtn.addEventListener('click', callInfer);

// Capture the satellite imagery within the bbox to use as the request image
captureSatBtn.addEventListener('click', () => {
  if (!bboxWgs84) { responseEl.textContent = 'Draw a bbox first'; return; }
  // Fit map to bbox to maximize resolution inside draw; then render a canvas snapshot
  const sw = L.latLng(bboxWgs84[0], bboxWgs84[1]);
  const ne = L.latLng(bboxWgs84[2], bboxWgs84[3]);
  const bounds = L.latLngBounds(sw, ne);
  imgInfo.textContent = 'Capturing satellite...';
  map.fitBounds(bounds, { animate: false });
  // Give tiles a moment to load
  setTimeout(() => {
    try {
      (window.leafletImage || leafletImage)(map, (err, canvas) => {
        if (err || !canvas) {
          imgInfo.textContent = 'Capture error: ' + (err?.message || err || 'unknown');
          return;
        }
        try {
          const mapSize = map.getSize();
          const tl = map.latLngToContainerPoint(ne); // top-left
          const br = map.latLngToContainerPoint(sw); // bottom-right
          const x = Math.max(0, Math.min(tl.x, br.x));
          const y = Math.max(0, Math.min(tl.y, br.y));
          const w = Math.min(mapSize.x - x, Math.abs(br.x - tl.x));
          const h = Math.min(mapSize.y - y, Math.abs(br.y - tl.y));
          const out = document.createElement('canvas');
          out.width = Math.max(1, Math.floor(w));
          out.height = Math.max(1, Math.floor(h));
          const ctx = out.getContext('2d');
          ctx.drawImage(canvas, x, y, w, h, 0, 0, out.width, out.height);
          const dataUrl = out.toDataURL('image/png'); // may throw if canvas tainted
          imageBase64 = dataUrl.split(',')[1];
          imgInfo.textContent = `Captured satellite ${out.width}x${out.height}`;
        } catch (e) {
          console.error('toDataURL error', e);
          imgInfo.textContent = 'Capture blocked by browser CORS. Switch basemap to OSM or provide a satellite tile with CORS support.';
          imageBase64 = null;
        }
      });
    } catch (e) {
      console.error('leafletImage error', e);
      imgInfo.textContent = 'Capture error: leaflet-image not available.';
    }
  }, 500);
});


