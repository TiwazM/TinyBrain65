"""Give the candidate its own disk/folder name; preserve the published V6 UI."""
from pathlib import Path
import hashlib,json
from tokenizer import ROOT

def main():
    for name in ['d81.py','run_xemu.py','fat_image.py','verify_xemu.py','native/build.bat','native/main.c']:
        p=ROOT/name;s=p.read_text();s=s.replace('TINYBRG6','TINYBR61').replace('tinybrg6','tinybr61')
        if name=='native/main.c':s=s.replace('V6 * trust, but verify','V6.1 * trust, but verify').replace('screen_write(22,55,','screen_write(22,53,')
        if name=='run_xemu.py':s=s.replace('TinyBrain65 / V6:', 'TinyBrain65 / V6.1 candidate:')
        p.write_text(s)
    baseline=ROOT.parent/'tinybrain65-v6-glow-brighter'
    unchanged=['build/GLOW.BIN','make_glow.py','make_title.py','native/m65_io.h','native/sidlm_core.c','native/kern_m65.c','native/kern_m65.s','native/kern_c.c','native/m65_float.c','native/float_m65.s','native/dma_m65.s','native/m65_far.h','native/tokenizer.c','native/tokens.h','native/qa.h','native/easter_eggs.c','data/tokenizer.json']
    hashes={}
    for name in unchanged:
        assert (ROOT/name).read_bytes()==(baseline/name).read_bytes(),name
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (ROOT/'build/runtime-unchanged.json').write_text(json.dumps(hashes,indent=2))
    print('Candidate disk/folder TINYBR61; unchanged inference, tokenizer and glow verified.')
if __name__=='__main__':main()
