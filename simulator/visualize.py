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
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;color:#ecf0f1;font-family:'Segoe UI',monospace;display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:10px}
#top{display:flex;gap:4px;align-items:center;padding:6px 16px;background:#1a1a2e;border-radius:8px;margin-bottom:6px;font-size:13px;min-width:780px;justify-content:center}
.tm{font-weight:bold;padding:2px 8px;border-radius:4px}
.tm-b{background:#1a5276;color:#5dade2}
.tm-r{background:#922b21;color:#ec7063}
#phase{color:#f39c12;margin:0 8px}
#winner{color:#2ecc71;font-weight:bold;margin-left:8px}
#main{display:flex;gap:8px;align-items:flex-start}
.side{width:140px;background:#1a1a2e;border-radius:8px;padding:8px;font-size:11px}
.side h3{font-size:12px;margin-bottom:4px;text-align:center}
.crown{font-size:18px;text-align:center;margin:2px 0}
.ex-bar{height:8px;background:#333;border-radius:4px;margin:2px 0}
.ex-fill{height:100%;border-radius:4px;transition:width 0.05s}
.ex-b{background:linear-gradient(90deg,#8e44ad,#9b59b6)}
.ex-r{background:linear-gradient(90deg,#c0392b,#e74c3c)}
.hand-card{display:inline-block;background:#2c3e50;border-radius:3px;padding:1px 4px;margin:1px;font-size:9px}
.nxt-card{color:#7f8c8d;font-size:9px}
canvas{border-radius:8px;border:2px solid #2c3e50}
#controls{display:flex;align-items:center;gap:8px;padding:6px 12px;background:#1a1a2e;border-radius:8px;margin-top:6px;min-width:780px}
#controls button{background:#2c3e50;color:#ecf0f1;border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:13px}
#controls button:hover{background:#34495e}
#controls button.active{background:#2980b9}
#seek{flex:1;height:6px;cursor:pointer;accent-color:#3498db}
#speed-sel{background:#2c3e50;color:#ecf0f1;border:none;border-radius:4px;padding:3px 6px;font-size:12px}
#finfo{font-size:11px;color:#95a5a6;white-space:nowrap}
#events{min-width:780px;max-width:780px;background:#1a1a2e;border-radius:8px;padding:6px 10px;margin-top:6px;font-size:10px;max-height:80px;overflow-y:auto}
#events div{padding:1px 0;border-bottom:1px solid #2c3e5033}
</style></head><body>
<div id="top">
<span id="time">0:00</span>
<span id="phase">[regulation]</span>
<span class="tm tm-b">BLUE</span><span id="b-cr">0</span>
<span style="color:#95a5a6">vs</span>
<span id="r-cr">0</span><span class="tm tm-r">RED</span>
<span id="winner"></span>
</div>
<div id="main">
<div class="side" id="blue-info">
<h3 style="color:#3498db">BLUE</h3>
<div class="crown" id="b-crown">0</div>
<div style="font-size:10px">Elixir: <span id="b-ex">5.0</span>/10</div>
<div class="ex-bar"><div class="ex-fill ex-b" id="b-ex-bar" style="width:50%"></div></div>
<div style="margin-top:4px;font-size:10px">Hand:</div>
<div id="b-hand"></div>
<div id="b-nxt" class="nxt-card"></div>
</div>
<canvas id="arena" width="450" height="800"></canvas>
<div class="side" id="red-info">
<h3 style="color:#e74c3c">RED</h3>
<div class="crown" id="r-crown">0</div>
<div style="font-size:10px">Elixir: <span id="r-ex">5.0</span>/10</div>
<div class="ex-bar"><div class="ex-fill ex-r" id="r-ex-bar" style="width:50%"></div></div>
<div style="margin-top:4px;font-size:10px">Hand:</div>
<div id="r-hand"></div>
<div id="r-nxt" class="nxt-card"></div>
</div>
</div>
<div id="controls">
<button id="btn-prev" title="Previous frame (Left arrow)">&#9664;&#9664;</button>
<button id="btn-play" title="Play/Pause (Space)">&#9654;</button>
<button id="btn-next" title="Next frame (Right arrow)">&#9654;&#9654;</button>
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
<script>
const SNAPS=__SNAPS_JSON__;
const CW=450,CH=800;
const SX=CW/18,SY=CH/32;
const CLR={blue:'#3498db',red:'#e74c3c'};
const CLR_D={blue:'#1a5276',red:'#922b21'};
const TWR=[
{tm:'blue',tp:'king',cx:8.5,cy:2.5,w:4,h:4},
{tm:'blue',tp:'princess',cx:3.0,cy:6.5,w:3,h:3},
{tm:'blue',tp:'princess',cx:14.0,cy:6.5,w:3,h:3},
{tm:'red',tp:'king',cx:8.5,cy:28.5,w:4,h:4},
{tm:'red',tp:'princess',cx:3.0,cy:24.5,w:3,h:3},
{tm:'red',tp:'princess',cx:14.0,cy:24.5,w:3,h:3}
];
let fi=0,playing=false,spd=1,timer=null;
const cv=document.getElementById('arena');
const ctx=cv.getContext('2d');
const seek=document.getElementById('seek');
seek.max=SNAPS.length-1;
function hpColor(f){
let r=Math.min(1,2*(1-f)),g=Math.min(1,2*f);
return `rgb(${Math.floor(r*255)},${Math.floor(g*255)},38)`;
}
function roundRect(x,y,w,h,r){
ctx.beginPath();
ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);
ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);
ctx.closePath();
}
function drawArena(){
ctx.fillStyle='#2d5016';ctx.fillRect(0,0,CW,CH);
ctx.fillStyle='rgba(41,128,185,0.7)';ctx.fillRect(0,15*SY,CW,2*SY);
ctx.fillStyle='#8B6914';ctx.strokeStyle='#5a4510';ctx.lineWidth=1;
[[3,6],[12,15]].forEach(b=>{
ctx.fillRect(b[0]*SX,15*SY,(b[1]-b[0])*SX,2*SY);
ctx.strokeRect(b[0]*SX,15*SY,(b[1]-b[0])*SX,2*SY);
});
ctx.strokeStyle='rgba(26,58,10,0.3)';ctx.lineWidth=0.5;
for(let x=0;x<=18;x++){ctx.beginPath();ctx.moveTo(x*SX,0);ctx.lineTo(x*SX,CH);ctx.stroke();}
for(let y=0;y<=32;y++){ctx.beginPath();ctx.moveTo(0,y*SY);ctx.lineTo(CW,y*SY);ctx.stroke();}
ctx.setLineDash([4,4]);ctx.strokeStyle='rgba(26,58,10,0.4)';ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(9*SX,0);ctx.lineTo(9*SX,CH);ctx.stroke();
ctx.setLineDash([]);
}
function drawTowers(snap){
const tw=snap.towers;
TWR.forEach((t,i)=>{
const td=tw[i]||null;
let fc,ec,a;
if(td&&!td.alive){fc='#555555';ec='#333333';a=0.5;}
else{fc=CLR[t.tm];ec=CLR_D[t.tm];a=0.85;}
const px=(t.cx-t.w/2)*SX,py=(t.cy-t.h/2)*SY;
const pw=t.w*SX,ph=t.h*SY;
ctx.globalAlpha=a;
ctx.fillStyle=fc;ctx.strokeStyle=ec;ctx.lineWidth=2;
roundRect(px,py,pw,ph,6);ctx.fill();ctx.stroke();
ctx.globalAlpha=1;
ctx.fillStyle='white';ctx.font='bold 14px monospace';ctx.textAlign='center';ctx.textBaseline='middle';
ctx.fillText(t.tp==='king'?'K':'P',t.cx*SX,t.cy*SY);
if(td){
const fr=td.max_hp>0?td.hp/td.max_hp:0;
const bw=t.w*SX*0.8,bx=t.cx*SX-bw/2,by=(t.cy+t.h/2)*SY+2;
ctx.fillStyle='#333';ctx.fillRect(bx,by,bw,5);
ctx.fillStyle=hpColor(fr);ctx.fillRect(bx,by,bw*fr,5);
ctx.fillStyle='white';ctx.font='8px monospace';ctx.textAlign='center';ctx.textBaseline='top';
ctx.fillText(td.hp+'/'+td.max_hp,t.cx*SX,by+6);
}
});
}
function drawTroops(snap){
snap.troops.forEach(u=>{
if(!u.alive)return;
const x=u.x*SX,y=u.y*SY;
const fc=CLR[u.team],ec=CLR_D[u.team];
const mhp=Math.max(u.max_hp,1);
const sz=Math.max(4,Math.min(14,Math.sqrt(mhp/30)*4));
if(u.is_building){
const bs=Math.max(SX*1.2,Math.min(SX*2.5,mhp/500*SX));
ctx.fillStyle=fc;ctx.strokeStyle=ec;ctx.lineWidth=1.5;
roundRect(x-bs/2,y-bs/2,bs,bs,3);ctx.fill();ctx.stroke();
ctx.fillStyle='white';ctx.font='bold 7px monospace';ctx.textAlign='center';ctx.textBaseline='middle';
ctx.fillText(u.name.substring(0,5),x,y);
}else if(u.transport==='Air'){
ctx.fillStyle=fc;ctx.strokeStyle=ec;ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(x,y-sz);ctx.lineTo(x+sz,y);ctx.lineTo(x,y+sz);ctx.lineTo(x-sz,y);ctx.closePath();
ctx.fill();ctx.stroke();
}else{
ctx.fillStyle=fc;ctx.strokeStyle=ec;ctx.lineWidth=1;
ctx.beginPath();ctx.arc(x,y,sz,0,Math.PI*2);ctx.fill();ctx.stroke();
}
const nm=u.name.substring(0,8);
ctx.font='bold 7px monospace';ctx.textAlign='center';ctx.textBaseline='bottom';
const tw=ctx.measureText(nm).width;
const lx=x-tw/2-2,ly=y-(u.is_building?sz+12:sz+8);
ctx.globalAlpha=0.7;ctx.fillStyle=fc;
roundRect(lx,ly,tw+4,10,2);ctx.fill();
ctx.globalAlpha=1;ctx.fillStyle='white';
ctx.fillText(nm,x,ly+9);
const fr=u.max_hp>0?u.hp/u.max_hp:0;
const bw=SX*1.0,bx=x-bw/2,by=y+(u.is_building?sz/2+2:sz+2);
ctx.fillStyle='#333';ctx.fillRect(bx,by,bw,3);
ctx.fillStyle=hpColor(fr);ctx.fillRect(bx,by,bw*fr,3);
});
}
function drawSpells(snap){
(snap.spells||[]).forEach(sp=>{
if(!sp.active)return;
const x=sp.x*SX,y=sp.y*SY,r=sp.radius*SX;
const fc=sp.team==='blue'?'rgba(93,173,226,0.25)':sp.team==='red'?'rgba(236,112,99,0.25)':'rgba(243,156,18,0.25)';
const ec=sp.team==='blue'?'#2e86c1':sp.team==='red'?'#cb4335':'#d68910';
ctx.fillStyle=fc;ctx.strokeStyle=ec;ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.stroke();
ctx.setLineDash([]);
ctx.fillStyle=ec;ctx.font='bold 8px monospace';ctx.textAlign='center';ctx.textBaseline='middle';
ctx.fillText(sp.name.substring(0,6),x,y);
});
}
function render(idx){
if(idx<0||idx>=SNAPS.length)return;
fi=idx;
const snap=SNAPS[fi];
ctx.clearRect(0,0,CW,CH);
drawArena();drawTowers(snap);drawSpells(snap);drawTroops(snap);
const t=snap.t;
const mn=Math.floor(t/60),sc=Math.floor(t%60);
document.getElementById('time').textContent=mn+':'+(sc<10?'0':'')+sc;
document.getElementById('phase').textContent='['+snap.phase+']';
const w=snap.winner;
document.getElementById('winner').textContent=w?'WINNER: '+w.toUpperCase():'';
['blue','red'].forEach(tm=>{
const p=tm[0];
const d=snap[tm];
document.getElementById(p+'-cr').textContent=d.crowns;
document.getElementById(p+'-crown').textContent=d.crowns;
document.getElementById(p+'-ex').textContent=d.elixir.toFixed(1);
document.getElementById(p+'-ex-bar').style.width=(d.elixir/10*100)+'%';
const hd=document.getElementById(p+'-hand');
hd.innerHTML=(d.hand||[]).map(c=>'<span class="hand-card">'+c+'</span>').join('');
const nd=document.getElementById(p+'-nxt');
nd.textContent=d.nxt?'Next: '+d.nxt+(d.nxt_cd>0?' ('+d.nxt_cd.toFixed(1)+'s)':''):'';
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
}
function play(){
if(playing)return;
playing=true;
document.getElementById('btn-play').innerHTML='&#9646;&#9646;';
const int=100/spd;
timer=setInterval(()=>{
if(fi>=SNAPS.length-1){stop();return;}
render(fi+1);
},int);
}
function stop(){
playing=false;
document.getElementById('btn-play').innerHTML='&#9654;';
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
