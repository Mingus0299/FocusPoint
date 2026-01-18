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

let points = []; // display coordinates (canvas space)
let closed = false;
let mode = 'draw'; // or 'view'
let tracking = false;
let demoInterval = null;

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
	if(!video.videoWidth || !video.videoHeight) return alert('Video not loaded');
	const sx = video.videoWidth / canvas.width;
	const sy = video.videoHeight / canvas.height;
	const framePoints = points.map(p=>({x: Math.round(p.x * sx), y: Math.round(p.y * sy)}));
	const payload = {frameWidth: video.videoWidth, frameHeight: video.videoHeight, polygon: framePoints};
	const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
	const a = document.createElement('a');
	a.href = URL.createObjectURL(blob);
	a.download = 'polygon.json';
	a.click();
});

function startDemoTracking(){
	if(tracking){
		tracking=false; startTrackBtn.textContent='Start Demo Tracking';
		clearInterval(demoInterval); demoInterval=null; return;
	}
	if(points.length < 3) return alert('Draw and finalize polygon first');
	// demo: jitter polygon vertices slightly and show confidence
	tracking = true; startTrackBtn.textContent='Stop Demo Tracking';
	demoInterval = setInterval(()=>{
		// add small random jitter to simulate motion
		for(let p of points){ p.x += (Math.random()-0.5)*2; p.y += (Math.random()-0.5)*2; }
		draw();
			// show confidence text at centroid
			const cx = points.reduce((s,p)=>s+p.x,0)/points.length;
			const cy = points.reduce((s,p)=>s+p.y,0)/points.length - 10;
			const conf = (0.5 + 0.5*Math.random()).toFixed(2);
			ctx.fillStyle = colors.confBg;
			ctx.fillRect(cx-42, cy-18, 84, 24);
			ctx.fillStyle = colors.infoText;
			ctx.font = '14px sans-serif';
			ctx.fillText('conf: ' + conf, cx-30, cy+2);
	}, 100);
}

startTrackBtn.addEventListener('click', startDemoTracking);

// initial resize in case a poster is present
setTimeout(()=> resizeCanvas(), 200);
