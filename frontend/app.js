const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const colors = {
  node: '#edba97',
  edge: '#e17426',
  edgeFill: 'rgba(225,116,38,0.12)',
  nodeStroke: '#701b15',
  infoText: 'rgb(164,192,193)',
  confBg: 'rgba(10,65,82,0.7)'
};

const fileInput = document.getElementById('fileInput');

const modeToggle = document.getElementById('modeToggle');
const loopToggle = document.getElementById('loopToggle');

const startAnnotBtn = document.getElementById('startAnnotBtn');
const undoBtn = document.getElementById('undoBtn');
const clearBtn = document.getElementById('clearBtn');

const startTrackBtn = document.getElementById('startTrackBtn');
const saveBtn = document.getElementById('saveBtn');

const info = document.getElementById('info');
const summaryText = document.getElementById('summaryText');

const layout = document.getElementById('layout');
const collapseBtn = document.getElementById('collapseBtn');
const sidebarOpenBtn = document.getElementById('sidebarOpenBtn');

/* ========= State ========= */
let points = [];                // canvas coords: [{x,y}, ...]
let closed = false;

let mode = 'view';              // 'draw' or 'view'
let annotationActive = false;   // NEW: must press "Start Annotation"
let tracking = false;

let sessionId = null;
let annotationId = null;
let socket = null;
let apiBase = null;
let sessionVideo = null;
let trackingMeta = null;
let wsKeepAlive = null;
let lastSummary = null;
let currentVideo = { kind: 'none', id: null, url: null };
let lastFrameAt = null;
let lastFramePoints = null;
const leadSeconds = 0.05;

/* ========= Helpers ========= */
function getVideoContainer() {
  return document.getElementById("videoContainer");
}

/**
 * Returns how the actual video image fits inside the container (object-fit: contain):
 * - displayed image width/height (dw/dh)
 * - offsets (ox/oy) where the image starts inside the container (black bars)
 * - scale factors to map frame<->canvas
 */
function getDisplayMetrics() {
  const vc = getVideoContainer();
  const r = vc.getBoundingClientRect();

  const cw = Math.max(1, r.width);
  const ch = Math.max(1, r.height);

  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;

  const containerAR = cw / ch;
  const videoAR = vw / vh;

  let dw, dh, ox, oy;

  if (videoAR > containerAR) {
    // video is wider: fit width
    dw = cw;
    dh = cw / videoAR;
    ox = 0;
    oy = (ch - dh) / 2;
  } else {
    // video is taller: fit height
    dh = ch;
    dw = ch * videoAR;
    oy = 0;
    ox = (cw - dw) / 2;
  }

  return {
    cw, ch, vw, vh,
    dw, dh, ox, oy,
    sx: vw / dw,   // canvas/image -> frame
    sy: vh / dh
  };
}

function freezeOnFirstFrame() {
  try { video.pause(); } catch {}

  const seekToZero = () => {
    try {
      video.pause();
      if (video.seekable && video.seekable.length > 0) {
        video.currentTime = 0;
      }
    } catch {}
  };

  if (video.readyState >= 1) {
    seekToZero();
  } else {
    video.addEventListener('loadedmetadata', seekToZero, { once: true });
  }

  // Ensure canvas lines up once the frame is actually visible
  const onReady = () => {
    resizeCanvas();
    draw();
  };

  video.addEventListener('seeked', onReady, { once: true });
  video.addEventListener(
    'loadeddata',
    () => {
      if (!(video.seekable && video.seekable.length > 0)) onReady();
    },
    { once: true }
  );
}

function isInsideDisplayedVideo(x, y) {
  const m = getDisplayMetrics();
  return (x >= m.ox && x <= m.ox + m.dw && y >= m.oy && y <= m.oy + m.dh);
}

function setInfo(text){
  if(info) info.textContent = text;
}

function renderSummary(summary){
  if(!summaryText) return;
  if(!summary){
    summaryText.textContent = 'No summary yet.';
    return;
  }
  const lines = [
    `Tracks: ${summary.total_tracks ?? 0}`,
    `Avg confidence: ${Math.round((summary.avg_confidence || 0) * 100)}%`,
    `Reanchor ok: ${summary.reanchor_success ?? 0}`,
    `Reanchor fail: ${summary.reanchor_fail ?? 0}`,
    `Out of frame: ${summary.out_of_frame ?? 0}`,
    `Low conf: ${summary.low_confidence ?? 0}`,
  ];
  if(summary.last_mode) lines.push(`Last mode: ${summary.last_mode}`);
  summaryText.textContent = lines.join('\n');
}

function resizeCanvas() {
  const vc = getVideoContainer();
  const r = vc.getBoundingClientRect();

  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));

  canvas.width = w;
  canvas.height = h;
  draw();
}

function resetForNewVideo(){
  clearAll();
  annotationActive = false;
  mode = 'view';
  modeToggle.textContent = 'Mode: View';
  canvas.style.pointerEvents = 'none';
  sessionVideo = null;
  trackingMeta = null;
  lastSummary = null;
  currentVideo = { kind: 'none', id: null, url: null };
  lastFrameAt = null;
  lastFramePoints = null;
}

function setVideoSource(url){
  video.src = url;
  video.loop = true;
  video.play().catch(()=>{});
}

function toCanvasCoords(evt) {
  const r = canvas.getBoundingClientRect();
  return { x: evt.clientX - r.left, y: evt.clientY - r.top };
}

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(points.length === 0) return;

  // edges
  ctx.lineWidth = 2;
  ctx.strokeStyle = colors.edge;
  ctx.fillStyle = colors.edgeFill;

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for(let i=1;i<points.length;i++) ctx.lineTo(points[i].x, points[i].y);
  if(closed) ctx.closePath();
  ctx.stroke();
  if(closed) ctx.fill();

  // nodes
  for(const p of points){
    ctx.beginPath();
    ctx.fillStyle = colors.node;
    ctx.arc(p.x,p.y,6,0,Math.PI*2);
    ctx.fill();
    ctx.strokeStyle = colors.nodeStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // tracking meta badge
  if(trackingMeta && points.length > 0){
    const cx = points.reduce((s,p)=>s+p.x,0)/points.length;
    const cy = points.reduce((s,p)=>s+p.y,0)/points.length - 12;
    const confPct = Math.round((trackingMeta.confidence || 0) * 100);
    const text = `${trackingMeta.mode || 'flow'} ${confPct}%`;

    ctx.font = '12px sans-serif';
    const metrics = ctx.measureText(text);
    const pad = 6;
    const boxW = metrics.width + pad * 2;
    const boxH = 18;

    ctx.fillStyle = colors.confBg;
    ctx.fillRect(cx - boxW/2, cy - boxH, boxW, boxH);
    ctx.fillStyle = colors.infoText;
    ctx.fillText(text, cx - boxW/2 + pad, cy - 4);
  }
}

function clearAll(){
  points = [];
  closed = false;
  trackingMeta = null;
  draw();
}

function undo(){
  if(points.length === 0) return;
  points.pop();
  closed = points.length >= 3; // keep closed state consistent
  draw();
}

/* ========= Annotation gating ========= */
function startAnnotation(){
  if(tracking) return alert('Stop tracking before annotating.');
  if(currentVideo.kind !== 'upload') return alert('Upload a video first.');

  freezeOnFirstFrame();

  annotationActive = true;
  mode = 'draw';
  modeToggle.textContent = 'Mode: Draw';
  canvas.style.pointerEvents = 'auto';

  clearAll();
  setInfo('Annotation active: click to add points (3+ closes polygon). Drag points to adjust.');
}

/* ========= Coordinate conversions ========= */
function toFramePoints(canvasPoints) {
  const m = getDisplayMetrics();

  return canvasPoints.map((p) => {
    // translate into displayed-image coordinates (remove black bar offsets)
    const ix = (p.x - m.ox);
    const iy = (p.y - m.oy);

    // map to frame
    const fx = Math.round(ix * m.sx);
    const fy = Math.round(iy * m.sy);

    return [fx, fy];
  });
}


function fromFramePoints(framePoints) {
  const m = getDisplayMetrics();

  return framePoints.map((pt) => {
    const x = Array.isArray(pt) ? pt[0] : pt.x;
    const y = Array.isArray(pt) ? pt[1] : pt.y;

    // frame -> displayed-image coords
    const ix = x / m.sx;
    const iy = y / m.sy;

    // add offsets back into canvas space
    return { x: ix + m.ox, y: iy + m.oy };
  });
}


/* ========= WebSocket ========= */
function connectWebSocket(wsPath, baseUrl){
  const base = new URL(baseUrl);
  const proto = base.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${base.host}${wsPath}`;
  const ws = new WebSocket(wsUrl);

  ws.addEventListener('open', ()=>{
    ws.send('ready');
    wsKeepAlive = setInterval(()=>{
      if(ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 15000);
    setInfo('WebSocket connected');
  });

  ws.addEventListener('message', (event)=>{
    let payload;
    try { payload = JSON.parse(event.data); } catch { return; }
    if(!payload || !payload.points) return;

    const rawPoints = fromFramePoints(payload.points);
    const now = performance.now();
    let displayPoints = rawPoints;

    if(!video.paused && lastFramePoints && lastFramePoints.length === rawPoints.length && lastFrameAt){
      const dt = (now - lastFrameAt) / 1000;
      if(dt > 0){
        displayPoints = rawPoints.map((p, i)=>{
          const prev = lastFramePoints[i];
          const vx = (p.x - prev.x) / dt;
          const vy = (p.y - prev.y) / dt;
          return { x: p.x + vx * leadSeconds, y: p.y + vy * leadSeconds };
        });
      }
    }

    points = displayPoints;
    closed = true;

    trackingMeta = {
      confidence: payload.confidence,
      mode: payload.mode,
      events: payload.events || []
    };
    lastFrameAt = now;
    lastFramePoints = rawPoints;

    setInfo(`mode: ${payload.mode || 'flow'} | conf: ${Math.round((payload.confidence || 0) * 100)}%`);
    draw();
  });

  ws.addEventListener('close', ()=>{
    if(wsKeepAlive){ clearInterval(wsKeepAlive); wsKeepAlive = null; }
    setInfo('WebSocket closed');
  });

  ws.addEventListener('error', ()=> setInfo('WebSocket error'));

  return ws;
}

/* ========= Interactions (nodes + drag) ========= */
function nearestPointIndex(pos, maxDist=10){
  let best = -1;
  let bestD = maxDist;
  for(let i=0;i<points.length;i++){
    const d = Math.hypot(points[i].x-pos.x, points[i].y-pos.y);
    if(d < bestD){ bestD = d; best = i; }
  }
  return best;
}

let draggingIndex = -1;

canvas.addEventListener('click', (e)=>{
  if(mode !== 'draw') return;
  if(!annotationActive) return;

  const p = toCanvasCoords(e);
  if (!isInsideDisplayedVideo(p.x, p.y)) return; // ignore black bars

  // don't add if clicking existing node
  const idx = nearestPointIndex(p, 8);
  if(idx >= 0) return;

  points.push(p);

  // polygon is considered closed once we have 3+ points
  closed = points.length >= 3;

  if(points.length === 3){
    setInfo('Polygon closed (3+ points). Keep clicking to add vertices. Drag points to adjust.');
  }

  draw();
});


canvas.addEventListener('contextmenu', (e)=>{
  e.preventDefault();
  if(mode !== 'draw' || !annotationActive) return;
  const p = toCanvasCoords(e);
  const idx = nearestPointIndex(p, 10);
  if(idx >= 0){
    points.splice(idx, 1);
    closed = points.length >= 3;
    draw();
  }
});

canvas.addEventListener('mousedown', (e)=>{
  if(mode !== 'draw' || !annotationActive) return;
  if(e.button !== 0) return;

  const p = toCanvasCoords(e);
  const idx = nearestPointIndex(p, 8);
  if(idx >= 0){
    draggingIndex = idx;
    canvas.style.cursor = 'grabbing';
  }
});

canvas.addEventListener('mousemove', (e)=>{
  const pos = toCanvasCoords(e);
  const idx = nearestPointIndex(pos, 8);

  if(draggingIndex >= 0){
    points[draggingIndex].x = pos.x;
    points[draggingIndex].y = pos.y;
    draw();
    return;
  }

  if(mode === 'draw' && annotationActive){
    canvas.style.cursor = (idx >= 0) ? 'grab' : (closed ? 'default' : 'crosshair');
  } else {
    canvas.style.cursor = 'default';
  }
});

canvas.addEventListener('mouseup', ()=>{
  if(draggingIndex >= 0){
    draggingIndex = -1;
    canvas.style.cursor = 'default';
  }
});

canvas.addEventListener('mouseleave', ()=>{
  draggingIndex = -1;
  canvas.style.cursor = 'default';
});

/* Keyboard shortcuts */
window.addEventListener('keydown', (e)=>{
  if(e.key === 'Backspace'){
    e.preventDefault();
    undo();
  } else if(e.key === 'Escape'){
    clearAll();
  }
});

/* ========= UI Buttons ========= */
startAnnotBtn.addEventListener('click', startAnnotation);
undoBtn.addEventListener('click', undo);
clearBtn.addEventListener('click', clearAll);

/* Optional mode toggle still exists, but start annotation is the intended entry */
modeToggle.addEventListener('click', ()=>{
  mode = (mode === 'draw') ? 'view' : 'draw';
  modeToggle.textContent = 'Mode: ' + (mode === 'draw' ? 'Draw' : 'View');
  canvas.style.pointerEvents = (mode === 'draw') ? 'auto' : 'none';
});

/* ========= Video loading ========= */
async function uploadVideo(file){
  const base = getApiBase();
  const form = new FormData();
  form.append('file', file, file.name || 'video.mp4');

  let res;
  try{
    res = await fetch(`${base}/video/upload`, { method: 'POST', body: form });
  } catch {
    return { ok: false, error: 'Failed to reach backend for upload.' };
  }

  if(!res.ok){
    const msg = await res.text();
    return { ok: false, error: `Upload error: ${msg}` };
  }

  const data = await res.json();
  const url = new URL(data.url, base).toString();
  return { ok: true, videoId: data.video_id, url };
}

fileInput.addEventListener('change', async (e)=>{
  const f = e.target.files && e.target.files[0];
  if(!f) return;

  resetForNewVideo();
  setInfo('Uploading video to backend…');

  const result = await uploadVideo(f);
  if(!result.ok){
    setInfo(result.error || 'Upload failed.');
    return;
  }

  currentVideo = { kind: 'upload', id: result.videoId, url: result.url };
  setVideoSource(result.url);
  setInfo('Video uploaded. Click “Start Annotation” to begin.');
});

video.addEventListener("loadedmetadata", () => {
  resizeCanvas();
});
window.addEventListener("resize", resizeCanvas);

video.addEventListener('pause', ()=>{ pauseTrackingSession(); });
video.addEventListener('play', ()=>{ resumeTrackingSession(); });

/* Loop toggle */
if(loopToggle){
  loopToggle.addEventListener('click', ()=>{
    video.loop = !video.loop;
    loopToggle.textContent = 'Loop: ' + (video.loop ? 'On' : 'Off');
  });
}
window.addEventListener('resize', ()=> resizeCanvas());

/* ========= Save polygon ========= */
saveBtn.addEventListener('click', ()=>{
  if(points.length < 3) return alert('No polygon to save');

  const frameW = (sessionVideo && sessionVideo.width) || video.videoWidth;
  const frameH = (sessionVideo && sessionVideo.height) || video.videoHeight;
  if(!frameW || !frameH) return alert('Video not loaded');

  const sx = frameW / canvas.width;
  const sy = frameH / canvas.height;

  const framePoints = points.map(p=>({x: Math.round(p.x * sx), y: Math.round(p.y * sy)}));
  const payload = {frameWidth: frameW, frameHeight: frameH, polygon: framePoints};

  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'polygon.json';
  a.click();
});

/* ========= Tracking ========= */
function getApiBase(){
  if(apiBase) return apiBase;
  if(window.API_BASE) return window.API_BASE;
  if(window.location && window.location.origin && window.location.origin !== 'null'){
    return window.location.origin;
  }
  return 'http://127.0.0.1:8000';
}

async function pauseTrackingSession(){
  if(!tracking || !sessionId) return;
  const base = getApiBase();
  try{
    await fetch(`${base}/sessions/${sessionId}/pause`, { method: 'POST' });
  } catch {}
}

async function resumeTrackingSession(){
  if(!tracking || !sessionId) return;
  const base = getApiBase();
  try{
    await fetch(`${base}/sessions/${sessionId}/resume`, { method: 'POST' });
  } catch {}
}

async function startTracking(){
  if(tracking){
    tracking = false;
    startTrackBtn.textContent = 'Start Demo Tracking';
    trackingMeta = null;
    lastSummary = null;
    lastFrameAt = null;
    lastFramePoints = null;

    if(socket){
      socket.close();
      socket = null;
    }

    const endSessionId = sessionId;
    const endApiBase = apiBase;

    if(endSessionId){
      try{
        const endRes = await fetch(`${endApiBase}/sessions/${endSessionId}/end`, {method:'POST'});
        if(endRes.ok){
          lastSummary = await endRes.json();
          renderSummary(lastSummary);
        } else {
          renderSummary(null);
        }
      } catch {}
    }

    sessionId = null;
    annotationId = null;

    mode = 'view';
    modeToggle.textContent = 'Mode: View';
    canvas.style.pointerEvents = 'none';
    annotationActive = false;

    setInfo('Tracking stopped.');
    draw();
    return;
  }

  if(points.length < 3) return alert('Annotate first (3 points).');
  if(currentVideo.kind !== 'upload' || !currentVideo.id){
    return alert('Upload a video first.');
  }
  apiBase = getApiBase();

  const payload = { video_id: currentVideo.id };

  let sessionRes;
  try{
    sessionRes = await fetch(`${apiBase}/sessions`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
  } catch {
    return alert('Failed to reach backend. Check API base.');
  }

  if(!sessionRes.ok){
    const msg = await sessionRes.text();
    return alert(`Session error: ${msg}`);
  }

  const sessionData = await sessionRes.json();
  sessionId = sessionData.session_id;
  sessionVideo = sessionData.video;
  lastSummary = null;
  lastFrameAt = null;
  lastFramePoints = null;
  renderSummary(null);

  socket = connectWebSocket(sessionData.ws_url, apiBase);

  const framePoints = toFramePoints(points);
  const annotationRes = await fetch(`${apiBase}/sessions/${sessionId}/annotations`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({points: framePoints})
  });

  if(!annotationRes.ok){
    const msg = await annotationRes.text();
    return alert(`Annotation error: ${msg}`);
  }

  const annotationData = await annotationRes.json();
  annotationId = annotationData.annotation_id;

  tracking = true;
  video.play().catch(()=>{});
  startTrackBtn.textContent = 'Stop Tracking';

  mode = 'view';
  modeToggle.textContent = 'Mode: View';
  canvas.style.pointerEvents = 'none';

  setInfo(`Tracking session ${sessionId}`);
}

startTrackBtn.addEventListener('click', startTracking);

/* ========= Sidebar hide/show ========= */
function collapseSidebar(){
  layout.classList.add('collapsed');
  collapseBtn.textContent = '»';
  setTimeout(resizeCanvas, 80);
}

function openSidebar(){
  layout.classList.remove('collapsed');
  collapseBtn.textContent = '⟨';
  setTimeout(resizeCanvas, 80);
}

collapseBtn.addEventListener('click', ()=>{
  if(layout.classList.contains('collapsed')) openSidebar();
  else collapseSidebar();
});

sidebarOpenBtn.addEventListener('click', openSidebar);

/* Initial state */
modeToggle.textContent = 'Mode: View';
canvas.style.pointerEvents = 'none';
setTimeout(()=> resizeCanvas(), 200);
