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

let imageBase64 = null;
let bboxWgs84 = null; // [south, west, north, east]

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
    const body = {
      bbox: bboxWgs84,
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
      renderArtifacts(json);
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


