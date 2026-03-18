import csv,sys,os,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from replay_battles import replay_battle,load_meta,norm

DD="/Users/okezuebell/Downloads/drive-download-20260310T214350Z-3-001"

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--meta',default=os.path.join(DD,'all_battle_meta_data.csv'))
    ap.add_argument('--workers',default=os.path.join(DD,'all_worker_rows.csv'))
    ap.add_argument('--limit',type=int,default=100)
    ap.add_argument('--offset',type=int,default=0)
    ap.add_argument('--verbose',action='store_true')
    ap.add_argument('--mode',default='Ladder')
    ap.add_argument('--min-plays',type=int,default=8)
    args=ap.parse_args()

    print("Loading metadata...")
    meta=load_meta(args.meta)
    print(f"  {len(meta)} battles")
    if args.mode:
        meta={k:v for k,v in meta.items() if v.get('gameMode','')==args.mode}
        print(f"  {len(meta)} after mode filter ({args.mode})")

    print("Loading all placements for matching battles...")
    plays_map={}
    mset=set(meta.keys())
    with open(args.workers) as f:
        for r in csv.DictReader(f):
            bid=r['battle_id']
            if bid not in mset:continue
            try:x=float(r.get('x','0') or '0')
            except:continue
            try:y=float(r.get('y','0') or '0')
            except:continue
            if bid not in plays_map:plays_map[bid]=[]
            plays_map[bid].append({
                'time':int(float(r.get('time','0'))),'team':r.get('team','blue'),
                'card':r.get('card',''),'x':int(x),'y':int(y),
                'tile_x':x/1000.0,'tile_y':y/1000.0,
                'ability':int(r.get('ability','0') or '0'),
                'player_id':r.get('player_id','')
            })
    print(f"  {len(plays_map)} battles loaded")

    valid=[]
    for bid in sorted(plays_map.keys()):
        ps=plays_map[bid]
        bp=sum(1 for p in ps if p['team']=='blue' and p.get('ability',0)==0)
        rp=sum(1 for p in ps if p['team']=='red' and p.get('ability',0)==0)
        if bp>=args.min_plays and rp>=args.min_plays:valid.append(bid)
    print(f"  {len(valid)} battles with >={args.min_plays} plays/side")
    if args.offset:valid=valid[args.offset:]
    if args.limit>0:valid=valid[:args.limit]
    print(f"  Running {len(valid)} battles")
    print()

    wm=0;ce=0;c1=0;tot=0;errs=0
    t0=time.time()
    for i,bid in enumerate(valid):
        ps=plays_map[bid]
        outcome=meta[bid]
        t0tag=outcome.get('t0_tag','').lstrip('#')
        try:
            g,info=replay_battle(bid,list(ps),outcome,verbose=args.verbose,pid=t0tag)
            tot+=1
            if info['win_match']:wm+=1
            if info['crown_exact']:ce+=1
            if info['crown_close']:c1+=1
            if not info['win_match'] and not args.verbose:
                print(f"  {bid}: sim={info['stm']} {info['sim_bc']}-{info['sim_rc']}  actual={outcome['result']} {outcome['tc']}-{outcome['oc']}  [X]")
        except Exception as e:
            errs+=1
            if args.verbose:print(f"  {bid}: ERROR {e}")
        if (i+1)%50==0:
            el=time.time()-t0
            print(f"  [{i+1:5d}/{len(valid)}] wm={wm}/{tot} ({100*wm/max(tot,1):.1f}%) ce={ce}/{tot} ({100*ce/max(tot,1):.1f}%) c1={c1}/{tot} ({100*c1/max(tot,1):.1f}%) errs={errs} {el:.0f}s")

    el=time.time()-t0
    print()
    print(f"=== Results ({tot} battles, {errs} errors, {el:.0f}s) ===")
    print(f"Winner match: {wm}/{tot} ({100*wm/max(tot,1):.1f}%)")
    print(f"Crown exact:  {ce}/{tot} ({100*ce/max(tot,1):.1f}%)")
    print(f"Crown +/-1:   {c1}/{tot} ({100*c1/max(tot,1):.1f}%)")

if __name__=='__main__':
    main()
