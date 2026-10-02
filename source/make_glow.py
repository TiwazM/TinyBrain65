"""Bake the native VIC-IV background: vignette plus title/rule halos.

The runtime composites crisp font pixels into colour tiles as text changes.
The brighter halo remains visible with hardware CRT simulation enabled.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from make_title import FONT
ROOT=Path(__file__).parent
def blur(a,sx,sy):
    for axis,sigma in [(1,sx),(0,sy)]:
        radius=int(np.ceil(3*sigma));x=np.arange(-radius,radius+1)
        k=np.exp(-x*x/(2*sigma*sigma));k/=k.sum()
        a=np.apply_along_axis(lambda row:np.convolve(row,k,mode='same'),axis,a)
    return a
def main():
    h,w=200,640;mask=np.zeros((h,w),np.float32)
    for i,c in enumerate('TINYBRAIN65'):
        for y,row in enumerate(FONT[c]):
            for x,v in enumerate(row):
                if v=='1':mask[8+y*4:8+y*4+4,24+i*36+x*6:24+i*36+x*6+6]=1
    for y in [6*8+3,23*8+3]:mask[y,24:77*8]=1
    yy,xx=np.mgrid[:h,:w];rad=((xx-319.5)/320)**2+((yy-99.5)/100)**2
    vignette=np.clip(1-rad,0,1)**.8
    light=1+14*vignette+78*blur(mask,2.4,1.2)+20*blur(mask,7,3.5)
    pixels=(32+np.clip(np.rint(light),0,63)).astype(np.uint8)
    # Full-colour character memory is tile-major, 8 rows of 8 pixels per tile.
    tiled=pixels.reshape(25,8,80,8).transpose(0,2,1,3).tobytes()
    assert len(tiled)==128000
    (ROOT/'build/GLOW.BIN').write_bytes(tiled)
    report={'bytes':len(tiled),'sha256':hashlib.sha256(tiled).hexdigest(),
            'width':w,'height':h,'tile_bytes':64,'first_tile':4096,
            'palette_first':32,'palette_count':64,'scanlines':False,
            'runtime_filter':False,'foreground_text_blurred':False,
            'halo_strength':{'near':78,'wide':20},'vignette_strength':14}
    (ROOT/'build/glow-asset.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
