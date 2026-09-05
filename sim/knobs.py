# engine constants with no exact public source, fitted against recorded games by sim/calib.py; the defaults are the engine's values
K={'sight_slack':0.0,'load_carry':1.0,'sep_strength':1.0,'sep_iters':1,'bridge_blend':0.0,'splash_hitbox':1.0,
   'kb_scale':1.0,'death_stagger':0.0,'detour_look':99.0,'charge_scale':1.0,'kite_drop':0,'kite_slack':0.0}
B={'sight_slack':(-1.5,2.0),'load_carry':(0.0,1.0),'sep_strength':(0.0,1.0),'sep_iters':(1,3),
   'bridge_blend':(0.0,1.0),'splash_hitbox':(0.0,1.0),'kb_scale':(0.5,1.5),'death_stagger':(0.0,0.5),'detour_look':(1.0,99.0),'charge_scale':(0.5,1.5),
   'kite_drop':(0,1),'kite_slack':(0.0,2.0)}
D=dict(K)
def use(ov=None):
    K.update(D);K.update(ov or {})
