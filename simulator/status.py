class Status:
    def __init__(self,kind,dur,val=0):
        self.kind=kind;self.dur=dur;self.val=val
    def tick(self,dt):self.dur-=dt
    @property
    def expired(self):return self.dur<=0
