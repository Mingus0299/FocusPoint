const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

// color palette from user: #edba97, #e17426, #701b15, rgb(164,192,193), #0a4152
const colors = {
	node: '#edba97',
	edge: '#e17426',
	edgeFill: 'rgba(225,116,38,0.12)',
	nodeStroke: '#701b15',
	infoText: 'rgb(164,192,193)',
	confBg: 'rgba(10,65,82,0.7)'
};

const fileInput = document.getElementById('fileInput');
const streamUrl = document.getElementById('streamUrl');
const setStream = document.getElementById('setStream');
const modeToggle = document.getElementById('modeToggle');
const finalizeBtn = document.getElementById('finalizeBtn');
const undoBtn = document.getElementById('undoBtn');
const clearBtn = document.getElementById('clearBtn');
const startTrackBtn = document.getElementById('startTrackBtn');
const saveBtn = document.getElementById('saveBtn');
const loopToggle = document.getElementById('loopToggle');
const info = document.getElementById('info');

let points = []; // display coordinates (canvas space)
let closed = false;
let mode = 'draw'; // or 'view'
let tracking = false;
let sessionId = null;
let annotationId = null;
let socket = null;
let apiBase = null;
let sessionVideo = null;
let trackingMeta = null;
let wsKeepAlive = null;

function resizeCanvas(){
	const rect = video.getBoundingClientRect();
	canvas.style.width = rect.width + 'px';
	canvas.style.height = rect.height + 'px';
	canvas.width = Math.round(rect.width);
	canvas.height = Math.round(rect.height);
	draw();
}

function toCanvasCoords(evt){
	const r = canvas.getBoundingClientRect();
	return {x: evt.clientX - r.left, y: evt.clientY - r.top};
}

function draw(){
	ctx.clearRect(0,0,canvas.width,canvas.height);
	if(points.length===0) return;
	ctx.lineWidth = 2;
	ctx.strokeStyle = colors.edge;
	ctx.fillStyle = colors.edgeFill;

	// draw edges
	ctx.beginPath();
	ctx.moveTo(points[0].x, points[0].y);
	for(let i=1;i<points.length;i++) ctx.lineTo(points[i].x, points[i].y);
	if(closed) ctx.closePath();
	ctx.stroke();
	if(closed){ ctx.fill(); }

	// draw nodes
	for(let i=0;i<points.length;i++){
		const p = points[i];
		ctx.beginPath();
		ctx.fillStyle = colors.node;
		ctx.arc(p.x,p.y,6,0,Math.PI*2);
		ctx.fill();
		ctx.strokeStyle = colors.nodeStroke;
		ctx.lineWidth = 1.5;
		ctx.stroke();
	}

	if(trackingMeta){
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

function addPoint(p){
	if(closed) return;
	points.push(p);
	draw();
}

function undo(){
	if(points.length===0) return;
	points.pop();
	draw();
}

function clearAll(){
	points = [];
	closed = false;
	draw();
}

function finalize(){
	if(points.length < 3) return alert('Need at least 3 points to finalize');
	closed = true;
	draw();
}

function setInfo(text){
	if(info) info.textContent = text;
}

function getApiBase(){
	if(apiBase) return apiBase;
	if(window.API_BASE) return window.API_BASE;
	const streamValue = streamUrl.value.trim();
	if(streamValue){
		try{
			return new URL(streamValue, window.location.href).origin;
		}catch(e){}
	}
	return 'http://127.0.0.1:8000';
}

function toFramePoints(canvasPoints){
	const frameW = (sessionVideo && sessionVideo.width) || video.videoWidth || canvas.width;
	const frameH = (sessionVideo && sessionVideo.height) || video.videoHeight || canvas.height;
	const sx = frameW / canvas.width;
	const sy = frameH / canvas.height;
	return canvasPoints.map(p=>[Math.round(p.x * sx), Math.round(p.y * sy)]);
}

function fromFramePoints(framePoints){
	const frameW = (sessionVideo && sessionVideo.width) || video.videoWidth || canvas.width;
	const frameH = (sessionVideo && sessionVideo.height) || video.videoHeight || canvas.height;
	const sx = canvas.width / frameW;
	const sy = canvas.height / frameH;
	return framePoints.map(p=>{
		const x = Array.isArray(p) ? p[0] : p.x;
		const y = Array.isArray(p) ? p[1] : p.y;
		return {x: x * sx, y: y * sy};
	});
}

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
		try{
			payload = JSON.parse(event.data);
		}catch(e){
			return;
		}
		if(!payload || !payload.points) return;
		points = fromFramePoints(payload.points);
		closed = true;
		trackingMeta = {
			confidence: payload.confidence,
			mode: payload.mode,
			events: payload.events || []
		};
		setInfo(`mode: ${payload.mode || 'flow'} | conf: ${Math.round((payload.confidence || 0) * 100)}%`);
		draw();
	});

	ws.addEventListener('close', ()=>{
		if(wsKeepAlive){
			clearInterval(wsKeepAlive);
			wsKeepAlive = null;
		}
		setInfo('WebSocket closed');
	});

	ws.addEventListener('error', ()=>{
		setInfo('WebSocket error');
	});

	return ws;
}

function nearestPointIndex(pos, maxDist=10){
	let best = -1; let bestD = maxDist;
	for(let i=0;i<points.length;i++){
		const d = Math.hypot(points[i].x-pos.x, points[i].y-pos.y);
		if(d < bestD){ bestD = d; best = i; }
	}
	return best;
}

let draggingIndex = -1;
let hoverIndex = -1;

canvas.addEventListener('click', (e)=>{
	if(mode !== 'draw') return;
	const p = toCanvasCoords(e);
	// don't add if clicking on an existing node
	const idx = nearestPointIndex(p, 8);
	if(idx < 0) addPoint(p);
});

canvas.addEventListener('contextmenu', (e)=>{
	e.preventDefault();
	const p = toCanvasCoords(e);
	const idx = nearestPointIndex(p, 10);
	if(idx >= 0){ points.splice(idx,1); draw(); }
});

canvas.addEventListener('mousedown', (e)=>{
	if(mode !== 'draw') return;
	if(e.button !== 0) return; // left only
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
	hoverIndex = idx;
	if(draggingIndex >= 0){
		// move the dragged point to current cursor
		points[draggingIndex].x = pos.x;
		points[draggingIndex].y = pos.y;
		draw();
	} else {
		if(mode === 'draw') canvas.style.cursor = (idx >= 0) ? 'grab' : 'crosshair';
		else canvas.style.cursor = 'default';
	}
});

canvas.addEventListener('mouseup', (e)=>{
	if(draggingIndex >= 0){ draggingIndex = -1; canvas.style.cursor = 'grab'; }
});

canvas.addEventListener('mouseleave', ()=>{ if(draggingIndex >= 0) draggingIndex = -1; canvas.style.cursor = 'default'; });

// Touch support: single-touch drag or add
canvas.addEventListener('touchstart', (e)=>{
	if(mode !== 'draw') return;
	const t = e.touches[0];
	const p = {x: t.clientX - canvas.getBoundingClientRect().left, y: t.clientY - canvas.getBoundingClientRect().top};
	const idx = nearestPointIndex(p, 12);
	if(idx >= 0){ draggingIndex = idx; }
	else { addPoint(p); }
	e.preventDefault();
});

canvas.addEventListener('touchmove', (e)=>{
	if(draggingIndex < 0) return;
	const t = e.touches[0];
	const p = {x: t.clientX - canvas.getBoundingClientRect().left, y: t.clientY - canvas.getBoundingClientRect().top};
	points[draggingIndex].x = p.x;
	points[draggingIndex].y = p.y;
	draw();
	e.preventDefault();
});

canvas.addEventListener('touchend', (e)=>{ draggingIndex = -1; });

window.addEventListener('keydown', (e)=>{
	if(e.key === 'Enter') { finalize(); }
	else if(e.key === 'Backspace'){ e.preventDefault(); undo(); }
	else if(e.key === 'Escape'){ clearAll(); }
});

finalizeBtn.addEventListener('click', finalize);
undoBtn.addEventListener('click', undo);
clearBtn.addEventListener('click', clearAll);

modeToggle.addEventListener('click', ()=>{
	mode = mode === 'draw' ? 'view' : 'draw';
	modeToggle.textContent = 'Mode: ' + (mode==='draw'?'Draw':'View');
	canvas.style.pointerEvents = mode==='draw' ? 'auto' : 'none';
});

fileInput.addEventListener('change', (e)=>{
	const f = e.target.files && e.target.files[0];
	if(!f) return;
	video.src = URL.createObjectURL(f);
	video.loop = true;
	video.play().catch(()=>{});
});

setStream.addEventListener('click', ()=>{
	const url = streamUrl.value.trim();
	if(!url) return alert('Enter a stream or video URL');
	video.src = url;
	video.loop = true;
	video.play().catch(()=>{});
});

video.addEventListener('loadedmetadata', ()=>{
	// enable looping by default so video replays until user disables
	video.loop = true;
	resizeCanvas();
});

// Loop toggle button
if(loopToggle){
	loopToggle.addEventListener('click', ()=>{
		video.loop = !video.loop;
		loopToggle.textContent = 'Loop: ' + (video.loop ? 'On' : 'Off');
	});
}
window.addEventListener('resize', ()=> resizeCanvas());

saveBtn.addEventListener('click', ()=>{
	if(points.length===0) return alert('No polygon to save');
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

async function startTracking(){
	if(tracking){
		tracking = false;
		startTrackBtn.textContent = 'Start Tracking';
		trackingMeta = null;
		if(socket){
			socket.close();
			socket = null;
		}
		if(sessionId){
			try{
				await fetch(`${apiBase}/sessions/${sessionId}/end`, {method:'POST'});
			}catch(e){}
		}
		sessionId = null;
		annotationId = null;
		setInfo('Tracking stopped');
		draw();
		return;
	}

	if(points.length < 3) return alert('Draw and finalize polygon first');
	apiBase = getApiBase();

	let sessionRes;
	try{
		sessionRes = await fetch(`${apiBase}/sessions`, {
			method: 'POST',
			headers: {'Content-Type': 'application/json'},
			body: JSON.stringify({})
		});
	}catch(e){
		return alert('Failed to reach backend. Check API base.');
	}
	if(!sessionRes.ok){
		const msg = await sessionRes.text();
		return alert(`Session error: ${msg}`);
	}
	const sessionData = await sessionRes.json();
	sessionId = sessionData.session_id;
	sessionVideo = sessionData.video;
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
	startTrackBtn.textContent = 'Stop Tracking';
	mode = 'view';
	modeToggle.textContent = 'Mode: View';
	canvas.style.pointerEvents = 'none';
	setInfo(`Tracking session ${sessionId}`);
}

startTrackBtn.textContent = 'Start Tracking';
startTrackBtn.addEventListener('click', startTracking);

// initial resize in case a poster is present
setTimeout(()=> resizeCanvas(), 200);

// Sidebar resizing & collapse
const layout = document.getElementById('layout');
const splitter = document.getElementById('splitter');
const collapseBtn = document.getElementById('collapseBtn');
let isResizing = false;

// restore saved sidebar width
const saved = localStorage.getItem('sidebarWidth');
if(saved && layout) layout.style.gridTemplateColumns = saved + ' 1fr';

if(splitter){
	splitter.addEventListener('mousedown', (e)=>{
		if(layout.classList.contains('collapsed')) return;
		isResizing = true; document.body.style.cursor = 'col-resize';
	});

	window.addEventListener('mousemove', (e)=>{
		if(!isResizing) return;
		const rect = layout.getBoundingClientRect();
		let newW = Math.max(120, Math.min(e.clientX - rect.left, rect.width - 160));
		layout.style.gridTemplateColumns = `${newW}px 1fr`;
	});

	window.addEventListener('mouseup', ()=>{
		if(!isResizing) return;
		isResizing = false; document.body.style.cursor = '';
		// persist width
		const cols = window.getComputedStyle(layout).gridTemplateColumns.split(' ')[0];
		localStorage.setItem('sidebarWidth', parseInt(cols,10));
	});

	// touch support for splitter
	splitter.addEventListener('touchstart', (e)=>{ if(layout.classList.contains('collapsed')) return; isResizing=true; });
	window.addEventListener('touchmove', (e)=>{ if(!isResizing) return; const t = e.touches[0]; const rect = layout.getBoundingClientRect(); let newW = Math.max(120, Math.min(t.clientX - rect.left, rect.width - 160)); layout.style.gridTemplateColumns = `${newW}px 1fr`; });
	window.addEventListener('touchend', ()=>{ if(!isResizing) return; isResizing=false; const cols = window.getComputedStyle(layout).gridTemplateColumns.split(' ')[0]; localStorage.setItem('sidebarWidth', parseInt(cols,10)); });
}

if(collapseBtn){
	collapseBtn.addEventListener('click', ()=>{
		const collapsed = layout.classList.toggle('collapsed');
		if(collapsed){
			// store current width
			const cols = window.getComputedStyle(layout).gridTemplateColumns.split(' ')[0];
			localStorage.setItem('sidebarPrevWidth', parseInt(cols,10));
			layout.style.gridTemplateColumns = '64px 1fr';
			collapseBtn.textContent = '»';
		} else {
			const prev = localStorage.getItem('sidebarPrevWidth') || localStorage.getItem('sidebarWidth') || 300;
			layout.style.gridTemplateColumns = `${prev}px 1fr`;
			collapseBtn.textContent = '⟨';
		}
		// trigger resize of canvas after layout change
		setTimeout(resizeCanvas, 120);
	});
}
