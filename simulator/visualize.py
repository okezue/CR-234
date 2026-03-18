import sys,os,json,random,argparse,threading,webbrowser
from http.server import HTTPServer,SimpleHTTPRequestHandler
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from game import Game,Dummy,card_info

W,H=18,32
_FILLER=['knight','archers','fireball','zap','valkyrie','musketeer','baby_dragon','mini_pekka']
def _mk_deck(cards):
    dk=list(cards);fi=0
    while len(dk)<8:
        c=_FILLER[fi%len(_FILLER)]
        if c not in dk:dk.append(c)
        fi+=1
    return dk
def _force_hand(g,tm,card):
    dk=g.players[tm].deck
    if card in dk.hand:return
    if dk.nxt==card:
        dk.nxt=dk.q.pop(0) if dk.q else None
    elif card in dk.q:
        dk.q.remove(card)
    else:return
    if len(dk.hand)<4:dk.hand.append(card)
    else:dk.hand[0]=card

HTML_TEMPLATE=r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CR Sim Visualizer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#f8f9fa;color:#1a1a2e;font-family:'Inter',system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:16px}
#top{display:flex;gap:12px;align-items:center;padding:10px 24px;background:#fff;border-radius:12px;margin-bottom:10px;font-size:14px;min-width:800px;justify-content:center;box-shadow:0 1px 3px rgba(0,0,0,0.08);border:1px solid #e9ecef}
.tm{font-weight:600;padding:3px 12px;border-radius:6px;font-size:13px;letter-spacing:0.3px}
.tm-b{background:#e8f4fd;color:#1971c2}
.tm-r{background:#fde8e8;color:#c92a2a}
#phase{color:#868e96;font-family:'JetBrains Mono',monospace;font-size:12px}
#winner{color:#2b8a3e;font-weight:700;margin-left:8px;font-size:13px}
#main{display:flex;gap:12px;align-items:flex-start}
.side{width:160px;background:#fff;border-radius:12px;padding:12px;font-size:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);border:1px solid #e9ecef}
.side h3{font-size:13px;font-weight:600;margin-bottom:6px;text-align:center;letter-spacing:0.5px}
.crown{font-size:28px;text-align:center;margin:4px 0;font-weight:700;font-family:'JetBrains Mono',monospace}
.ex-label{font-size:11px;color:#868e96;margin-top:8px}
.ex-bar{height:6px;background:#e9ecef;border-radius:3px;margin:4px 0;overflow:hidden}
.ex-fill{height:100%;border-radius:3px;transition:width 0.08s ease}
.ex-b{background:linear-gradient(90deg,#4dabf7,#228be6)}
.ex-r{background:linear-gradient(90deg,#ff8787,#e03131)}
.hand-label{font-size:10px;color:#868e96;margin-top:10px;margin-bottom:4px}
.hand-card{display:inline-block;background:#f1f3f5;border:1px solid #dee2e6;border-radius:4px;padding:2px 6px;margin:2px;font-size:10px;font-family:'JetBrains Mono',monospace;cursor:pointer;transition:all 0.15s}
.hand-card:hover{background:#e8f4fd;border-color:#74c0fc}
.hand-card.sel{background:#d0ebff;border-color:#228be6;color:#1864ab}
.nxt-card{color:#adb5bd;font-size:10px;font-family:'JetBrains Mono',monospace;margin-top:4px}
.canvas-wrap{position:relative;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.12);border:1px solid #dee2e6}
canvas{display:block}
#controls{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#fff;border-radius:12px;margin-top:10px;min-width:800px;box-shadow:0 1px 3px rgba(0,0,0,0.08);border:1px solid #e9ecef}
#controls button{background:#f1f3f5;color:#495057;border:1px solid #dee2e6;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:14px;transition:all 0.15s;font-family:'Inter',system-ui,sans-serif}
#controls button:hover{background:#e9ecef;border-color:#adb5bd}
#controls button.active{background:#228be6;color:#fff;border-color:#1971c2}
#seek{flex:1;height:4px;cursor:pointer;accent-color:#228be6;-webkit-appearance:none;appearance:none;background:#e9ecef;border-radius:2px;outline:none}
#seek::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#228be6;cursor:pointer;border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,0.2)}
#speed-sel{background:#f1f3f5;color:#495057;border:1px solid #dee2e6;border-radius:6px;padding:4px 8px;font-size:12px;font-family:'Inter',system-ui,sans-serif}
#finfo{font-size:11px;color:#adb5bd;white-space:nowrap;font-family:'JetBrains Mono',monospace}
#events{min-width:800px;max-width:800px;background:#fff;border-radius:12px;padding:8px 14px;margin-top:10px;font-size:11px;max-height:90px;overflow-y:auto;box-shadow:0 1px 3px rgba(0,0,0,0.08);border:1px solid #e9ecef;font-family:'JetBrains Mono',monospace;color:#495057}
#events div{padding:2px 0;border-bottom:1px solid #f1f3f5}
#graphs{display:flex;gap:12px;min-width:800px;margin-top:10px}
.graph-box{flex:1;background:#fff;border-radius:12px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);border:1px solid #e9ecef}
.graph-box h4{font-size:11px;color:#868e96;margin-bottom:6px;font-weight:500;letter-spacing:0.5px;text-transform:uppercase}
.graph-box canvas{width:100%;height:100px}
.place-legend{font-size:10px;color:#868e96;text-align:center;margin-top:8px;font-style:italic}
</style></head><body>
<div id="top">
<span id="time" style="font-family:'JetBrains Mono',monospace;font-weight:600;font-size:16px">0:00</span>
<span id="phase">[regulation]</span>
<span class="tm tm-b">BLUE</span><span id="b-cr" style="font-family:'JetBrains Mono',monospace;font-weight:700;font-size:18px">0</span>
<span style="color:#adb5bd;font-size:12px">vs</span>
<span id="r-cr" style="font-family:'JetBrains Mono',monospace;font-weight:700;font-size:18px">0</span><span class="tm tm-r">RED</span>
<span id="winner"></span>
</div>
<div id="main">
<div class="side" id="blue-info">
<h3 style="color:#228be6">BLUE</h3>
<div class="crown" id="b-crown" style="color:#228be6">0</div>
<div class="ex-label">Elixir <span id="b-ex" style="font-family:'JetBrains Mono',monospace;color:#495057">5.0</span> / 10</div>
<div class="ex-bar"><div class="ex-fill ex-b" id="b-ex-bar" style="width:50%"></div></div>
<div class="hand-label">HAND</div>
<div id="b-hand"></div>
<div id="b-nxt" class="nxt-card"></div>
</div>
<div class="canvas-wrap">
<canvas id="arena" width="450" height="800"></canvas>
</div>
<div class="side" id="red-info">
<h3 style="color:#e03131">RED</h3>
<div class="crown" id="r-crown" style="color:#e03131">0</div>
<div class="ex-label">Elixir <span id="r-ex" style="font-family:'JetBrains Mono',monospace;color:#495057">5.0</span> / 10</div>
<div class="ex-bar"><div class="ex-fill ex-r" id="r-ex-bar" style="width:50%"></div></div>
<div class="hand-label">HAND</div>
<div id="r-hand"></div>
<div id="r-nxt" class="nxt-card"></div>
</div>
</div>
<div id="controls">
<button id="btn-prev" title="Previous (Left)">&#9664;&#9664;</button>
<button id="btn-play" title="Play/Pause (Space)">&#9654;</button>
<button id="btn-next" title="Next (Right)">&#9654;&#9654;</button>
<input type="range" id="seek" min="0" max="100" value="0">
<select id="speed-sel">
<option value="0.5">0.5x</option>
<option value="1" selected>1x</option>
<option value="2">2x</option>
<option value="4">4x</option>
<option value="8">8x</option>
</select>
<span id="finfo">F:0/0</span>
</div>
<div id="events"></div>
<div id="graphs">
<div class="graph-box"><h4>Elixir</h4><canvas id="g-elixir" height="100"></canvas></div>
<div class="graph-box"><h4>Tower HP</h4><canvas id="g-towers" height="100"></canvas></div>
</div>
<div class="place-legend">Click a card in hand to highlight valid deploy zones</div>
<script>
const SNAPS=__SNAPS_JSON__;
const CW=450,CH=800;
const SX=CW/18,SY=CH/32;
const B='#228be6',B2='#1864ab',BL='#d0ebff';
const R='#e03131',R2='#a61e1e',RL='#fde8e8';
const CLR={blue:B,red:R};
const CLR_L={blue:BL,red:RL};
const TWR=[
{tm:'blue',tp:'king',cx:8.5,cy:2.5,w:4,h:4},
{tm:'blue',tp:'princess',cx:3.0,cy:6.5,w:3,h:3},
{tm:'blue',tp:'princess',cx:14.0,cy:6.5,w:3,h:3},
{tm:'red',tp:'king',cx:8.5,cy:28.5,w:4,h:4},
{tm:'red',tp:'princess',cx:3.0,cy:24.5,w:3,h:3},
{tm:'red',tp:'princess',cx:14.0,cy:24.5,w:3,h:3}
];
let fi=0,playing=false,spd=1,timer=null,selTm=null;
const cv=document.getElementById('arena');
const ctx=cv.getContext('2d');
const seek=document.getElementById('seek');
seek.max=SNAPS.length-1;
function hpFrac(hp,mx){return mx>0?hp/mx:0;}
function hpColor(f){
if(f>0.6)return '#2b8a3e';
if(f>0.3)return '#e67700';
return '#c92a2a';
}
function rr(x,y,w,h,r){
ctx.beginPath();
ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);
ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);
ctx.closePath();
}
function drawArena(){
ctx.fillStyle='#e8f5e9';ctx.fillRect(0,0,CW,CH);
ctx.fillStyle='rgba(33,150,243,0.18)';ctx.fillRect(0,15*SY,CW,2*SY);
ctx.fillStyle='#d7ccc8';ctx.strokeStyle='#bcaaa4';ctx.lineWidth=1;
[[3,6],[12,15]].forEach(b=>{
rr(b[0]*SX,15*SY,(b[1]-b[0])*SX,2*SY,3);ctx.fill();ctx.stroke();
});
ctx.strokeStyle='rgba(0,0,0,0.06)';ctx.lineWidth=0.5;
for(let x=0;x<=18;x++){ctx.beginPath();ctx.moveTo(x*SX,0);ctx.lineTo(x*SX,CH);ctx.stroke();}
for(let y=0;y<=32;y++){ctx.beginPath();ctx.moveTo(0,y*SY);ctx.lineTo(CW,y*SY);ctx.stroke();}
ctx.setLineDash([6,6]);ctx.strokeStyle='rgba(0,0,0,0.08)';ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(9*SX,0);ctx.lineTo(9*SX,CH);ctx.stroke();
ctx.setLineDash([]);
}
function validZone(snap,tm){
let yMin,yMax;
if(tm==='blue'){yMin=0;yMax=15;
const lp=snap.towers[2],rp=snap.towers[1];
if(lp&&!lp.alive)yMax=17;if(rp&&!rp.alive)yMax=17;
}else{yMin=17;yMax=32;
const lp=snap.towers[4],rp=snap.towers[5];
if(lp&&!lp.alive)yMin=15;if(rp&&!rp.alive)yMin=15;
}
return{yMin,yMax};
}
function drawValidZone(snap){
if(!selTm)return;
const z=validZone(snap,selTm);
const c=selTm==='blue'?'rgba(34,139,230,0.12)':'rgba(224,49,49,0.12)';
const sc=selTm==='blue'?'rgba(34,139,230,0.35)':'rgba(224,49,49,0.35)';
ctx.fillStyle=c;
ctx.fillRect(0,z.yMin*SY,CW,(z.yMax-z.yMin)*SY);
ctx.strokeStyle=sc;ctx.lineWidth=2;ctx.setLineDash([8,4]);
ctx.strokeRect(1,z.yMin*SY+1,CW-2,(z.yMax-z.yMin)*SY-2);
ctx.setLineDash([]);
ctx.fillStyle=sc;ctx.font='600 11px Inter,system-ui';ctx.textAlign='center';ctx.textBaseline='top';
ctx.fillText('DEPLOY ZONE',CW/2,z.yMin*SY+4);
}
function drawTowers(snap){
const tw=snap.towers;
TWR.forEach((t,i)=>{
const td=tw[i]||null;
const dead=td&&!td.alive;
const px=(t.cx-t.w/2)*SX,py=(t.cy-t.h/2)*SY;
const pw=t.w*SX,ph=t.h*SY;
if(dead){
ctx.globalAlpha=0.25;ctx.fillStyle='#adb5bd';ctx.strokeStyle='#868e96';
}else{
ctx.globalAlpha=0.9;ctx.fillStyle=CLR_L[t.tm];ctx.strokeStyle=CLR[t.tm];
}
ctx.lineWidth=2;rr(px,py,pw,ph,8);ctx.fill();ctx.stroke();
ctx.globalAlpha=1;
ctx.fillStyle=dead?'#868e96':CLR[t.tm];
ctx.font='bold 16px "JetBrains Mono",monospace';ctx.textAlign='center';ctx.textBaseline='middle';
ctx.fillText(t.tp==='king'?'K':'P',t.cx*SX,t.cy*SY-2);
if(td&&!dead){
const fr=hpFrac(td.hp,td.max_hp);
const bw=t.w*SX*0.75,bx=t.cx*SX-bw/2,by=(t.cy+t.h/2)*SY+3;
ctx.fillStyle='#e9ecef';rr(bx,by,bw,5,2.5);ctx.fill();
ctx.fillStyle=hpColor(fr);rr(bx,by,bw*Math.max(fr,0.01),5,2.5);ctx.fill();
ctx.fillStyle='#495057';ctx.font='500 8px "JetBrains Mono",monospace';
ctx.textAlign='center';ctx.textBaseline='top';
ctx.fillText(Math.max(0,td.hp),t.cx*SX,by+7);
}
});
}
function drawTroops(snap){
snap.troops.forEach(u=>{
if(!u.alive)return;
const x=u.x*SX,y=u.y*SY;
const fc=CLR[u.team];
const mhp=Math.max(u.max_hp,1);
const sz=Math.max(5,Math.min(14,Math.sqrt(mhp/30)*4));
ctx.shadowColor='rgba(0,0,0,0.15)';ctx.shadowBlur=3;ctx.shadowOffsetY=1;
if(u.is_building){
const bs=Math.max(SX*1.2,Math.min(SX*2.5,mhp/500*SX));
ctx.fillStyle=CLR_L[u.team];ctx.strokeStyle=fc;ctx.lineWidth=1.5;
rr(x-bs/2,y-bs/2,bs,bs,4);ctx.fill();ctx.stroke();
ctx.shadowBlur=0;
ctx.fillStyle=fc;ctx.font='600 7px "JetBrains Mono",monospace';ctx.textAlign='center';ctx.textBaseline='middle';
ctx.fillText(u.name.substring(0,5),x,y);
}else if(u.transport==='Air'){
ctx.fillStyle=fc;ctx.strokeStyle='#fff';ctx.lineWidth=1.5;
ctx.beginPath();ctx.moveTo(x,y-sz);ctx.lineTo(x+sz*0.8,y+sz*0.4);ctx.lineTo(x-sz*0.8,y+sz*0.4);ctx.closePath();
ctx.fill();ctx.stroke();
}else{
ctx.fillStyle=fc;ctx.strokeStyle='#fff';ctx.lineWidth=1.5;
ctx.beginPath();ctx.arc(x,y,sz,0,Math.PI*2);ctx.fill();ctx.stroke();
}
ctx.shadowBlur=0;
const nm=u.name.substring(0,8);
ctx.font='500 7px "Inter",system-ui';ctx.textAlign='center';ctx.textBaseline='bottom';
const tw2=ctx.measureText(nm).width;
const lx=x-tw2/2-3,ly=y-(u.is_building?sz+14:sz+10);
ctx.globalAlpha=0.85;ctx.fillStyle='#fff';
rr(lx,ly,tw2+6,12,3);ctx.fill();
ctx.strokeStyle='rgba(0,0,0,0.08)';ctx.lineWidth=0.5;
rr(lx,ly,tw2+6,12,3);ctx.stroke();
ctx.globalAlpha=1;ctx.fillStyle='#343a40';
ctx.fillText(nm,x,ly+11);
const fr=hpFrac(u.hp,u.max_hp);
const bw=SX*1.0,bx=x-bw/2,by=y+(u.is_building?sz/2+3:sz+3);
ctx.fillStyle='#dee2e6';rr(bx,by,bw,3,1.5);ctx.fill();
ctx.fillStyle=hpColor(fr);rr(bx,by,bw*fr,3,1.5);ctx.fill();
});
}
function drawSpells(snap){
(snap.spells||[]).forEach(sp=>{
if(!sp.active)return;
const x=sp.x*SX,y=sp.y*SY,r=sp.radius*SX;
const base=sp.team==='blue'?[34,139,230]:sp.team==='red'?[224,49,49]:[230,180,34];
ctx.fillStyle=`rgba(${base},0.12)`;
ctx.strokeStyle=`rgba(${base},0.5)`;
ctx.lineWidth=1.5;ctx.setLineDash([5,3]);
ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.stroke();
ctx.setLineDash([]);
ctx.fillStyle=`rgba(${base},0.8)`;ctx.font='600 8px "JetBrains Mono",monospace';
ctx.textAlign='center';ctx.textBaseline='middle';
ctx.fillText(sp.name.substring(0,6),x,y);
});
}
function render(idx){
if(idx<0||idx>=SNAPS.length)return;
fi=idx;
const snap=SNAPS[fi];
ctx.clearRect(0,0,CW,CH);
drawArena();drawValidZone(snap);drawTowers(snap);drawSpells(snap);drawTroops(snap);
const t=snap.t;
const mn=Math.floor(t/60),sc=Math.floor(t%60);
document.getElementById('time').textContent=mn+':'+(sc<10?'0':'')+sc;
document.getElementById('phase').textContent='['+snap.phase+']';
const w=snap.winner;
document.getElementById('winner').textContent=w?w.toUpperCase()+' WINS':'';
['blue','red'].forEach(tm=>{
const p=tm[0];
const d=snap[tm];
document.getElementById(p+'-cr').textContent=d.crowns;
document.getElementById(p+'-crown').textContent=d.crowns;
document.getElementById(p+'-ex').textContent=d.elixir.toFixed(1);
document.getElementById(p+'-ex-bar').style.width=(d.elixir/10*100)+'%';
const hd=document.getElementById(p+'-hand');
hd.innerHTML=(d.hand||[]).map(c=>{
const sel=selTm===tm?' sel':'';
return '<span class="hand-card'+sel+'" data-tm="'+tm+'" data-card="'+c+'">'+c+'</span>';
}).join('');
hd.querySelectorAll('.hand-card').forEach(el=>{
el.addEventListener('click',()=>{
const clickTm=el.getAttribute('data-tm');
selTm=selTm===clickTm?null:clickTm;
render(fi);
});
});
const nd=document.getElementById(p+'-nxt');
nd.textContent=d.nxt?'next: '+d.nxt+(d.nxt_cd>0?' ('+d.nxt_cd.toFixed(1)+'s)':''):'';
});
seek.value=fi;
document.getElementById('finfo').textContent='F:'+fi+'/'+SNAPS.length+' T:'+t.toFixed(1)+'s';
const evDiv=document.getElementById('events');
if(snap.events&&snap.events.length){
snap.events.forEach(e=>{
const d=document.createElement('div');
d.textContent='['+t.toFixed(1)+'] '+e;
evDiv.appendChild(d);
});
evDiv.scrollTop=evDiv.scrollHeight;
}
drawGraphs();
}
function drawGraphs(){
drawElixirGraph();drawTowerGraph();
}
function drawElixirGraph(){
const cv2=document.getElementById('g-elixir');
const w2=cv2.clientWidth||380,h2=100;
cv2.width=w2;cv2.height=h2;
const c2=cv2.getContext('2d');
c2.fillStyle='#fff';c2.fillRect(0,0,w2,h2);
c2.strokeStyle='#f1f3f5';c2.lineWidth=0.5;
for(let i=0;i<=10;i+=2){const yy=h2-i/10*h2;c2.beginPath();c2.moveTo(0,yy);c2.lineTo(w2,yy);c2.stroke();}
const n=Math.min(fi+1,SNAPS.length);
if(n<2)return;
[{tm:'blue',c:B},{tm:'red',c:R}].forEach(({tm,c})=>{
c2.strokeStyle=c;c2.lineWidth=1.5;c2.globalAlpha=0.8;
c2.beginPath();
for(let i=0;i<n;i++){
const x=i/(SNAPS.length-1)*w2;
const y=h2-(SNAPS[i][tm]?.elixir||0)/10*h2;
i===0?c2.moveTo(x,y):c2.lineTo(x,y);
}
c2.stroke();c2.globalAlpha=1;
});
c2.fillStyle='#228be6';c2.globalAlpha=0.3;
c2.fillRect(fi/(SNAPS.length-1)*w2,0,2,h2);
c2.globalAlpha=1;
}
function drawTowerGraph(){
const cv2=document.getElementById('g-towers');
const w2=cv2.clientWidth||380,h2=100;
cv2.width=w2;cv2.height=h2;
const c2=cv2.getContext('2d');
c2.fillStyle='#fff';c2.fillRect(0,0,w2,h2);
c2.strokeStyle='#f1f3f5';c2.lineWidth=0.5;
for(let i=0;i<=4;i++){const yy=i/4*h2;c2.beginPath();c2.moveTo(0,yy);c2.lineTo(w2,yy);c2.stroke();}
const n=Math.min(fi+1,SNAPS.length);
if(n<2)return;
[{idx:0,c:B,d:[4,4]},{idx:1,c:B,d:[]},{idx:2,c:B,d:[2,2]},
{idx:3,c:R,d:[4,4]},{idx:4,c:R,d:[]},{idx:5,c:R,d:[2,2]}].forEach(({idx,c,d})=>{
c2.strokeStyle=c;c2.lineWidth=1.2;c2.globalAlpha=0.6;c2.setLineDash(d);
c2.beginPath();
let mx=0;
for(let i=0;i<n;i++){
const tw=SNAPS[i].towers[idx];
if(tw&&tw.max_hp>mx)mx=tw.max_hp;
}
if(mx===0){c2.setLineDash([]);return;}
for(let i=0;i<n;i++){
const x=i/(SNAPS.length-1)*w2;
const tw=SNAPS[i].towers[idx];
const y=h2-(tw?tw.hp/mx:0)*h2;
i===0?c2.moveTo(x,y):c2.lineTo(x,y);
}
c2.stroke();c2.setLineDash([]);c2.globalAlpha=1;
});
c2.fillStyle='#228be6';c2.globalAlpha=0.3;
c2.fillRect(fi/(SNAPS.length-1)*w2,0,2,h2);
c2.globalAlpha=1;
}
function play(){
if(playing)return;
playing=true;
document.getElementById('btn-play').textContent='\u23F8';
const int=100/spd;
timer=setInterval(()=>{
if(fi>=SNAPS.length-1){stop();return;}
render(fi+1);
},int);
}
function stop(){
playing=false;
document.getElementById('btn-play').textContent='\u25B6';
if(timer){clearInterval(timer);timer=null;}
}
function togglePlay(){if(playing)stop();else play();}
document.getElementById('btn-play').addEventListener('click',togglePlay);
document.getElementById('btn-prev').addEventListener('click',()=>{stop();render(Math.max(0,fi-1));});
document.getElementById('btn-next').addEventListener('click',()=>{stop();render(Math.min(SNAPS.length-1,fi+1));});
seek.addEventListener('input',()=>{stop();render(parseInt(seek.value));});
document.getElementById('speed-sel').addEventListener('change',e=>{
spd=parseFloat(e.target.value);
if(playing){stop();play();}
});
document.addEventListener('keydown',e=>{
if(e.code==='Space'){e.preventDefault();togglePlay();}
else if(e.code==='ArrowLeft'){stop();render(Math.max(0,fi-1));}
else if(e.code==='ArrowRight'){stop();render(Math.min(SNAPS.length-1,fi+1));}
else if(e.code==='Escape'){selTm=null;render(fi);}
});
render(0);
</script></body></html>'''

def visualize(game,port=0):
    snaps=game.replay.snaps
    if not snaps:
        print("No replay snaps.");return
    sj=json.dumps(snaps)
    html=HTML_TEMPLATE.replace('__SNAPS_JSON__',sj)
    class H(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        def log_message(self,*a):pass
    srv=HTTPServer(('127.0.0.1',port),H)
    p=srv.server_address[1]
    t=threading.Thread(target=srv.serve_forever,daemon=True)
    t.start()
    url=f'http://127.0.0.1:{p}'
    print(f"Visualizer at {url}")
    webbrowser.open(url)
    try:input("Press Enter to stop server...")
    except KeyboardInterrupt:pass
    srv.shutdown()

MULTI_TEMPLATE=r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CR Sim — Battle Browser</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#f8f9fa;color:#1a1a2e;font-family:'Inter',system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
#sidebar{width:320px;background:#fff;border-right:1px solid #e9ecef;display:flex;flex-direction:column;flex-shrink:0}
#sb-head{padding:16px;border-bottom:1px solid #e9ecef}
#sb-head h2{font-size:15px;font-weight:700;color:#228be6;letter-spacing:0.5px}
#sb-head p{font-size:11px;color:#868e96;margin-top:2px}
#sb-filter{display:flex;gap:6px;padding:10px 16px;border-bottom:1px solid #e9ecef}
#sb-filter button{flex:1;padding:5px 0;font-size:11px;border:1px solid #dee2e6;border-radius:6px;background:#fff;color:#495057;cursor:pointer;font-family:'Inter',system-ui;transition:all 0.15s}
#sb-filter button:hover{background:#f1f3f5}
#sb-filter button.active{background:#228be6;color:#fff;border-color:#1971c2}
#sb-list{flex:1;overflow-y:auto;padding:8px}
.battle-card{padding:10px 12px;border-radius:8px;margin-bottom:6px;cursor:pointer;border:1px solid #e9ecef;transition:all 0.15s}
.battle-card:hover{background:#f8f9fa;border-color:#adb5bd}
.battle-card.sel{background:#e7f5ff;border-color:#228be6}
.bc-id{font-family:'JetBrains Mono',monospace;font-size:10px;color:#adb5bd}
.bc-score{display:flex;align-items:center;gap:8px;margin-top:4px}
.bc-sim{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600}
.bc-vs{font-size:10px;color:#adb5bd}
.bc-actual{font-family:'JetBrains Mono',monospace;font-size:12px;color:#868e96}
.bc-tags{display:flex;gap:4px;margin-top:4px;flex-wrap:wrap}
.tag{font-size:9px;padding:1px 6px;border-radius:4px;font-weight:500}
.tag-win{background:#d3f9d8;color:#2b8a3e}
.tag-loss{background:#ffe3e3;color:#c92a2a}
.tag-exact{background:#d0ebff;color:#1864ab}
.tag-close{background:#fff3bf;color:#e67700}
.bc-decks{font-size:9px;color:#adb5bd;margin-top:4px;font-family:'JetBrains Mono',monospace;line-height:1.4}
#viewer{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto}
#viewer iframe{border:none;width:100%;height:100%}
#placeholder{text-align:center;color:#adb5bd}
#placeholder h3{font-size:16px;font-weight:500;margin-bottom:4px}
#placeholder p{font-size:12px}
#sb-stats{padding:10px 16px;border-top:1px solid #e9ecef;font-size:11px;color:#868e96;font-family:'JetBrains Mono',monospace}
</style></head><body>
<div id="sidebar">
<div id="sb-head"><h2>BATTLE BROWSER</h2><p id="sb-count"></p></div>
<div id="sb-filter">
<button class="active" data-f="all">All</button>
<button data-f="correct">Correct</button>
<button data-f="wrong">Wrong</button>
<button data-f="exact">Exact</button>
</div>
<div id="sb-list"></div>
<div id="sb-stats"></div>
</div>
<div id="viewer">
<div id="placeholder"><h3>Select a battle</h3><p>Click a battle on the left to view</p></div>
</div>
<script>
const GAMES=__GAMES_JSON__;
const SNAPS_MAP=__SNAPS_MAP__;
let curFilter='all',curBid=null;
function filterGames(f){
curFilter=f;
document.querySelectorAll('#sb-filter button').forEach(b=>{
b.classList.toggle('active',b.getAttribute('data-f')===f);
});
renderList();
}
function renderList(){
const list=document.getElementById('sb-list');
list.innerHTML='';
const filtered=GAMES.filter(g=>{
if(curFilter==='correct')return g.win_match;
if(curFilter==='wrong')return !g.win_match;
if(curFilter==='exact')return g.crown_exact;
return true;
});
filtered.forEach(g=>{
const d=document.createElement('div');
d.className='battle-card'+(curBid===g.bid?' sel':'');
const sw=g.sim_bc>g.sim_rc?'blue':g.sim_bc<g.sim_rc?'red':'draw';
const sc=sw==='blue'?'color:#228be6':sw==='red'?'color:#e03131':'color:#868e96';
let tags='';
if(g.win_match)tags+='<span class="tag tag-win">WIN</span>';
else tags+='<span class="tag tag-loss">MISS</span>';
if(g.crown_exact)tags+='<span class="tag tag-exact">EXACT</span>';
else if(g.crown_close)tags+='<span class="tag tag-close">~1</span>';
const bd=(g.b_deck||[]).slice(0,4).join(', ');
const rd=(g.r_deck||[]).slice(0,4).join(', ');
d.innerHTML=`<div class="bc-id">${g.bid}</div>
<div class="bc-score">
<span class="bc-sim" style="${sc}">SIM ${g.sim_bc}-${g.sim_rc}</span>
<span class="bc-vs">vs</span>
<span class="bc-actual">REAL ${g.actual_bc}-${g.actual_rc}</span>
</div>
<div class="bc-tags">${tags}</div>
<div class="bc-decks">B: ${bd}...<br>R: ${rd}...</div>`;
d.addEventListener('click',()=>loadBattle(g.bid));
list.appendChild(d);
});
document.getElementById('sb-count').textContent=`${filtered.length} / ${GAMES.length} battles`;
const wm=GAMES.filter(g=>g.win_match).length;
const ce=GAMES.filter(g=>g.crown_exact).length;
document.getElementById('sb-stats').textContent=`Win: ${wm}/${GAMES.length} (${(100*wm/GAMES.length).toFixed(1)}%) | Exact: ${ce}/${GAMES.length} (${(100*ce/GAMES.length).toFixed(1)}%)`;
}
function loadBattle(bid){
curBid=bid;
renderList();
document.getElementById('viewer').innerHTML=`<iframe src="/battle/${bid}"></iframe>`;
}
document.querySelectorAll('#sb-filter button').forEach(b=>{
b.addEventListener('click',()=>filterGames(b.getAttribute('data-f')));
});
renderList();
</script></body></html>'''

def visualize_multi_lazy(battle_list,outcomes,placements,pids,port=0):
    import time as _time
    from replay_battles import replay_battle
    print(f"Running stats for {len(battle_list)} battles...")
    glist=[];valid_bids=set()
    done=0;wm=0;ce=0
    for b in battle_list:
        bid=b['bid']
        if bid not in outcomes or bid not in placements:continue
        g,info=replay_battle(bid,placements[bid],outcomes[bid],pid=pids.get(bid))
        entry={k:v for k,v in b.items()}
        entry.update({'sim_bc':info['sim_bc'],'sim_rc':info['sim_rc'],
            'actual_bc':info['actual_bc'],'actual_rc':info['actual_rc'],
            'win_match':info['win_match'],'crown_exact':info['crown_exact'],
            'crown_close':abs(info['sim_bc']-info['actual_bc'])<=1 and abs(info['sim_rc']-info['actual_rc'])<=1})
        glist.append(entry);valid_bids.add(bid)
        if info['win_match']:wm+=1
        if info['crown_exact']:ce+=1
        done+=1
        if done%100==0:print(f"  [{done}/{len(battle_list)}] wm={wm}/{done} ({100*wm/done:.1f}%)")
    print(f"Done. {len(glist)} battles. Win={wm}/{done} ({100*wm/max(1,done):.1f}%) Exact={ce}/{done} ({100*ce/max(1,done):.1f}%)")
    index_html=MULTI_TEMPLATE.replace('__GAMES_JSON__',json.dumps(glist))
    index_html=index_html.replace('__SNAPS_MAP__','{}')
    index_html=index_html.replace('__VIEWER_HTML__','""')
    class H(SimpleHTTPRequestHandler):
        def do_GET(self):
            path=self.path.split('?')[0]
            if path.startswith('/battle/'):
                bid=path[8:]
                if bid in valid_bids:
                    g2,_=replay_battle(bid,placements[bid],outcomes[bid],pid=pids.get(bid))
                    bhtml=HTML_TEMPLATE.replace('__SNAPS_JSON__',json.dumps(g2.replay.snaps))
                    self.send_response(200)
                    self.send_header('Content-Type','text/html')
                    self.end_headers()
                    self.wfile.write(bhtml.encode())
                    return
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.end_headers()
            self.wfile.write(index_html.encode())
        def log_message(self,*a):pass
    srv=HTTPServer(('127.0.0.1',port),H)
    p=srv.server_address[1]
    t=threading.Thread(target=srv.serve_forever,daemon=True)
    t.start()
    url=f'http://127.0.0.1:{p}'
    print(f"\nMulti-battle visualizer at {url}")
    webbrowser.open(url)
    try:
        input("Press Enter to stop server...")
    except (KeyboardInterrupt,EOFError):
        print("Server running. Ctrl+C to stop.")
        try:
            while True:_time.sleep(1)
        except KeyboardInterrupt:pass
    srv.shutdown()

BROWSER_TEMPLATE=r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CR Battle Browser</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#f8f9fa;color:#1a1a2e;font-family:'Inter',system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
#sidebar{width:360px;background:#fff;border-right:1px solid #e9ecef;display:flex;flex-direction:column;flex-shrink:0}
#sb-head{padding:14px 16px;border-bottom:1px solid #e9ecef}
#sb-head h2{font-size:15px;font-weight:700;color:#228be6;letter-spacing:0.5px}
#sb-count{font-size:11px;color:#868e96;margin-top:2px}
.filter-row{display:flex;gap:4px;padding:6px 16px}
.filter-row:last-of-type{border-bottom:1px solid #e9ecef;padding-bottom:10px}
.filter-row button{flex:1;padding:4px 0;font-size:10px;border:1px solid #dee2e6;border-radius:5px;background:#fff;color:#495057;cursor:pointer;font-family:'Inter',system-ui;transition:all 0.12s}
.filter-row button:hover{background:#f1f3f5}
.filter-row button.active{background:#228be6;color:#fff;border-color:#1971c2}
.filter-row button.active-green{background:#2b8a3e;color:#fff;border-color:#237032}
.filter-row button.active-red{background:#c92a2a;color:#fff;border-color:#a61e1e}
#sb-search{padding:6px 16px;border-bottom:1px solid #e9ecef}
#sb-search input{width:100%;padding:6px 10px;border:1px solid #dee2e6;border-radius:6px;font-size:12px;font-family:'JetBrains Mono',monospace;outline:none}
#sb-search input:focus{border-color:#228be6}
#sb-list{flex:1;overflow-y:auto;padding:6px}
.bc{padding:8px 10px;border-radius:8px;margin-bottom:3px;cursor:pointer;border:1px solid #e9ecef;transition:all 0.12s;display:flex;align-items:center;gap:8px}
.bc:hover{background:#f8f9fa;border-color:#adb5bd}
.bc.sel{background:#e7f5ff;border-color:#228be6}
.bc-result{font-size:18px;font-weight:700;width:24px;text-align:center;font-family:'JetBrains Mono',monospace;flex-shrink:0}
.bc-w{color:#2b8a3e}.bc-l{color:#c92a2a}.bc-d{color:#868e96}
.bc-body{flex:1;min-width:0}
.bc-top{display:flex;align-items:center;gap:6px}
.bc-id{font-family:'JetBrains Mono',monospace;font-size:9px;color:#adb5bd}
.bc-score{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#495057}
.bc-sim{font-family:'JetBrains Mono',monospace;font-size:11px;color:#868e96}
.bc-decks{font-size:9px;color:#adb5bd;font-family:'JetBrains Mono',monospace;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bc-right{display:flex;flex-direction:column;align-items:flex-end;gap:2px;flex-shrink:0}
.bc-lvl{font-size:10px;color:#868e96;font-family:'JetBrains Mono',monospace}
.bc-acc{font-size:9px;padding:1px 5px;border-radius:3px;font-weight:600}
.acc-yes{background:#d3f9d8;color:#2b8a3e}
.acc-no{background:#ffe3e3;color:#c92a2a}
.acc-exact{background:#d0ebff;color:#1864ab}
#sb-stats{padding:8px 16px;border-top:1px solid #e9ecef;font-size:10px;color:#868e96;font-family:'JetBrains Mono',monospace;line-height:1.5}
#viewer{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto;background:#f1f3f5}
#viewer iframe{border:none;width:100%;height:100%}
#placeholder{text-align:center;color:#adb5bd}
#placeholder h3{font-size:18px;font-weight:500;margin-bottom:6px}
#placeholder p{font-size:13px}
#loading{display:none;text-align:center;color:#228be6;font-size:13px;padding:40px}
</style></head><body>
<div id="sidebar">
<div id="sb-head"><h2>BATTLE BROWSER</h2><div id="sb-count"></div></div>
<div class="filter-row" id="f-result">
<button class="active" data-f="all">All</button>
<button data-f="W">Wins</button>
<button data-f="L">Losses</button>
<button data-f="D">Draws</button>
</div>
<div class="filter-row" id="f-acc">
<button class="active" data-a="all">All Sim</button>
<button data-a="correct">Correct</button>
<button data-a="wrong">Wrong</button>
<button data-a="exact">Exact Crown</button>
</div>
<div id="sb-search"><input type="text" id="search" placeholder="Search battle ID or card..."></div>
<div id="sb-list"></div>
<div id="sb-stats"></div>
</div>
<div id="viewer">
<div id="placeholder"><h3>Select a battle</h3><p>Click any battle to simulate and view</p><p style="margin-top:8px;font-size:11px;color:#ced4da">Battles replayed on-demand (~1s)</p></div>
<div id="loading">Replaying battle...</div>
</div>
<script>
const GAMES=__GAMES_JSON__;
let fResult='all',fAcc='all',curBid=null,searchQ='';
const PAGE=60;let page=0;
const hasSim=GAMES.length>0&&GAMES[0].sim_bc!==undefined;
function filtered(){
return GAMES.filter(g=>{
if(fResult!=='all'&&g.result!==fResult)return false;
if(hasSim){
if(fAcc==='correct'&&!g.win_match)return false;
if(fAcc==='wrong'&&g.win_match)return false;
if(fAcc==='exact'&&!g.crown_exact)return false;
}
if(searchQ){
const q=searchQ;
if(!g.bid.toLowerCase().includes(q)&&
!(g.b_deck||[]).join(' ').includes(q)&&
!(g.r_deck||[]).join(' ').includes(q))return false;
}
return true;
});
}
function renderList(){
const list=document.getElementById('sb-list');
list.innerHTML='';
const f=filtered();
const shown=f.slice(0,(page+1)*PAGE);
shown.forEach(g=>{
const d=document.createElement('div');
d.className='bc'+(curBid===g.bid?' sel':'');
const rc=g.result==='W'?'bc-w':g.result==='L'?'bc-l':'bc-d';
const bd=(g.b_deck||[]).slice(0,4).join(', ');
const rd=(g.r_deck||[]).slice(0,4).join(', ');
let simHtml='';
let accHtml='';
if(hasSim){
const sw=g.sim_bc>g.sim_rc?'B':g.sim_bc<g.sim_rc?'R':'=';
simHtml=`<span class="bc-sim">sim ${g.sim_bc}-${g.sim_rc}</span>`;
if(g.crown_exact)accHtml='<span class="bc-acc acc-exact">EXACT</span>';
else if(g.win_match)accHtml='<span class="bc-acc acc-yes">OK</span>';
else accHtml='<span class="bc-acc acc-no">MISS</span>';
}
d.innerHTML=`<div class="bc-result ${rc}">${g.result}</div>
<div class="bc-body">
<div class="bc-top"><span class="bc-score">${g.tc}-${g.oc}</span>${simHtml}<span class="bc-id">${g.bid.slice(-8)}</span></div>
<div class="bc-decks">${bd}</div>
<div class="bc-decks">${rd}</div>
</div>
<div class="bc-right">
<div class="bc-lvl">${g.b_klvl}v${g.r_klvl}</div>
${accHtml}
</div>`;
d.addEventListener('click',()=>loadBattle(g.bid));
list.appendChild(d);
});
if(shown.length<f.length){
const more=document.createElement('div');
more.style.cssText='text-align:center;padding:8px;color:#228be6;cursor:pointer;font-size:11px';
more.textContent=`+ ${f.length-shown.length} more`;
more.addEventListener('click',()=>{page++;renderList();});
list.appendChild(more);
}
document.getElementById('sb-count').textContent=f.length+' / '+GAMES.length+' battles';
const ws=GAMES.filter(g=>g.result==='W').length;
const ls=GAMES.filter(g=>g.result==='L').length;
let stats=`W:${ws} L:${ls} D:${GAMES.length-ws-ls}`;
if(hasSim){
const wm=GAMES.filter(g=>g.win_match).length;
const ce=GAMES.filter(g=>g.crown_exact).length;
stats+=`\nSim: ${wm}/${GAMES.length} correct (${(100*wm/GAMES.length).toFixed(1)}%)`;
stats+=` | ${ce} exact (${(100*ce/GAMES.length).toFixed(1)}%)`;
}
document.getElementById('sb-stats').textContent=stats;
}
function loadBattle(bid){
curBid=bid;
renderList();
document.getElementById('placeholder').style.display='none';
document.getElementById('loading').style.display='block';
const ex=document.querySelector('#viewer iframe');
if(ex)ex.remove();
const iframe=document.createElement('iframe');
iframe.src='/battle/'+bid;
iframe.onload=()=>{document.getElementById('loading').style.display='none';};
document.getElementById('viewer').appendChild(iframe);
}
document.querySelectorAll('#f-result button').forEach(b=>{
b.addEventListener('click',()=>{
fResult=b.getAttribute('data-f');
document.querySelectorAll('#f-result button').forEach(x=>x.classList.remove('active'));
b.classList.add('active');
page=0;renderList();
});
});
document.querySelectorAll('#f-acc button').forEach(b=>{
b.addEventListener('click',()=>{
fAcc=b.getAttribute('data-a');
document.querySelectorAll('#f-acc button').forEach(x=>x.classList.remove('active'));
b.classList.add('active');
page=0;renderList();
});
});
document.getElementById('search').addEventListener('input',e=>{
searchQ=e.target.value.toLowerCase();page=0;renderList();
});
if(!hasSim)document.getElementById('f-acc').style.display='none';
renderList();
</script></body></html>'''

def visualize_browser(battle_list,outcomes,placements,pids,port=0):
    import time as _time
    from replay_battles import replay_battle
    print(f"{len(battle_list)} battles in browser")
    valid_bids=set(b['bid'] for b in battle_list)
    index_html=BROWSER_TEMPLATE.replace('__GAMES_JSON__',json.dumps(battle_list))
    class H(SimpleHTTPRequestHandler):
        def do_GET(self):
            path=self.path.split('?')[0]
            if path.startswith('/battle/'):
                bid=path[8:]
                if bid in valid_bids and bid in placements and bid in outcomes:
                    g,_=replay_battle(bid,placements[bid],outcomes[bid],pid=pids.get(bid))
                    bhtml=HTML_TEMPLATE.replace('__SNAPS_JSON__',json.dumps(g.replay.snaps))
                    self.send_response(200)
                    self.send_header('Content-Type','text/html')
                    self.end_headers()
                    self.wfile.write(bhtml.encode())
                    return
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.end_headers()
            self.wfile.write(index_html.encode())
        def log_message(self,*a):pass
    srv=HTTPServer(('127.0.0.1',port),H)
    p=srv.server_address[1]
    t=threading.Thread(target=srv.serve_forever,daemon=True)
    t.start()
    url=f'http://127.0.0.1:{p}'
    print(f"Battle browser at {url} ({len(battle_list)} battles)")
    webbrowser.open(url)
    try:
        input("Press Enter to stop server...")
    except (KeyboardInterrupt,EOFError):
        print("Server running. Ctrl+C to stop.")
        try:
            while True:_time.sleep(1)
        except KeyboardInterrupt:pass
    srv.shutdown()

def scn_pekka_push():
    random.seed(42)
    dk=_mk_deck(['pekka'])
    g=Game(p1={'deck':dk,'drag_del':0.3,'drag_std':0})
    _force_hand(g,'blue','pekka')
    g.players['blue'].elixir=10
    g.play_card('blue','pekka',3,14)
    g.run(30)
    return g
def scn_mk_v_skarmy():
    random.seed(42)
    dk_r=_mk_deck(['skeleton_army'])
    dk_b=_mk_deck(['mega_knight'])
    g=Game(p1={'deck':dk_b,'drag_del':0.0,'drag_std':0},
           p2={'deck':dk_r,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'red','skeleton_army')
    _force_hand(g,'blue','mega_knight')
    g.players['red'].elixir=10;g.players['blue'].elixir=10
    g.play_card('red','skeleton_army',9,17)
    g.run(5)
    g.players['blue'].elixir=10
    g.play_card('blue','mega_knight',9,14)
    g.run(10)
    return g
def scn_hog_push():
    random.seed(42)
    dk=_mk_deck(['hog_rider'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','hog_rider')
    g.players['blue'].elixir=10
    g.play_card('blue','hog_rider',3,14)
    g.run(20)
    return g
def scn_full_match():
    random.seed(42)
    dk_b=_mk_deck(['pekka','knight','archers','fireball'])
    dk_r=_mk_deck(['mega_knight','valkyrie','musketeer','zap'])
    g=Game(p1={'deck':dk_b,'drag_del':0.4,'drag_std':0},
           p2={'deck':dk_r,'drag_del':0.4,'drag_std':0})
    for _ in range(3000):
        g.tick()
        if g.ended:break
        for tm in ('blue','red'):
            p=g.players[tm]
            if p.deck and p.deck.hand and p.elixir>=5:
                c=p.deck.hand[0]
                ci=card_info(c)
                if p.elixir>=ci['cost']:
                    if tm=='blue':x,y=9,12
                    else:x,y=9,20
                    g.play_card(tm,c,x,y)
    return g
def scn_nw_bats():
    random.seed(42)
    dk=_mk_deck(['night_witch'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','night_witch')
    g.players['blue'].elixir=10
    g.play_card('blue','night_witch',9,5)
    g.run(10)
    return g
def scn_rage_hog():
    random.seed(42)
    dk=_mk_deck(['hog_rider','rage'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','hog_rider')
    g.players['blue'].elixir=10
    g.play_card('blue','hog_rider',3,14)
    g.run(5)
    hogs=[t for t in g.players['blue'].troops if t.alive and t.name=='Hog Rider']
    if hogs:
        _force_hand(g,'blue','rage')
        g.players['blue'].elixir=10
        g.play_card('blue','rage',hogs[0].x,hogs[0].y)
    g.run(15)
    return g
def scn_furnace():
    random.seed(42)
    dk=_mk_deck(['furnace'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','furnace')
    g.players['blue'].elixir=10
    g.play_card('blue','furnace',5,10)
    g.run(15)
    return g

SCENARIOS={
    'pekka_push':scn_pekka_push,
    'mk_v_skarmy':scn_mk_v_skarmy,
    'hog_push':scn_hog_push,
    'full_match':scn_full_match,
    'nw_bats':scn_nw_bats,
    'rage_hog':scn_rage_hog,
    'furnace':scn_furnace,
}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Clash Royale web visualizer')
    ap.add_argument('scenario',nargs='?',default='pekka_push',
                    choices=list(SCENARIOS.keys()),help='scenario to visualize')
    ap.add_argument('--port',type=int,default=0,help='server port (0=auto)')
    args=ap.parse_args()
    print(f"Running scenario: {args.scenario}")
    g=SCENARIOS[args.scenario]()
    print(f"Replay: {len(g.replay.snaps)} snaps, T={g.t:.1f}s")
    visualize(g,port=args.port)
