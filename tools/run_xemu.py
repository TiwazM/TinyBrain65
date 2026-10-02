"""Launch the packaged TinyBrain65 demo with a private copy of the user's SD image."""
from pathlib import Path
import argparse,hashlib,os,shutil,subprocess,sys
from fat_image import FAT

def prepare(base,destination,model,glow):
    base=Path(base).resolve();destination=Path(destination).resolve()
    if base==destination or not base.is_file() or base.suffix.lower()!='.img':
        raise RuntimeError('Choose an existing Xemu .img as the base, separate from the private copy.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        if shutil.disk_usage(destination.parent).free<base.stat().st_size+100_000_000:
            raise RuntimeError('Not enough free space for the private SD image (usually 4 GB).')
        pending=destination.with_name('preparing-'+destination.name)
        print('Copying your Xemu SD image for this demo. The original stays unchanged.',flush=True)
        shutil.copyfile(base,pending);fat=FAT(pending,True)
        try:
            for name,data in [('MEGAQA.BIN',model),('GLOW.BIN',glow)]:
                if name not in {v[0] for v in fat.entries()}:fat.add(name,data)
                if fat.read(name)!=data:raise RuntimeError('The base image contains a different '+name+'; choose a base without that file.')
        finally:fat.close()
        pending.rename(destination)
    fat=FAT(destination)
    try:
        if fat.read('MEGAQA.BIN')!=model:raise RuntimeError('Private SD model mismatch.')
        if fat.read('GLOW.BIN')!=glow:raise RuntimeError('Private SD artwork mismatch.')
    finally:fat.close()

def main():
    home=Path(__file__).resolve().parents[1];app=Path(os.environ.get('APPDATA',''))/'xemu-lgb/mega65'
    ap=argparse.ArgumentParser();ap.add_argument('--fast',action='store_true');ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--xemu',default=r'C:\Program Files\xemu\xmega65.exe');ap.add_argument('--rom',type=Path,default=app/'MEGA65.ROM');ap.add_argument('--base-image',type=Path,default=app/'mega65.img');args=ap.parse_args()
    disk=home/'SD-CARD/TINYBR61/TINYBR61.D81';model=(home/'SD-CARD/TINYBR61/MEGAQA.BIN').read_bytes()
    glow=(home/'SD-CARD/TINYBR61/GLOW.BIN').read_bytes()
    image=home/'xemu-data'/('megaqa-'+hashlib.sha256(model+glow).hexdigest()[:12]+'.img')
    if not args.rom.is_file() or not Path(args.xemu).is_file():raise RuntimeError('Set --rom and --xemu to your installed ROM and emulator.')
    prepare(args.base_image,image,model,glow)
    if args.prepare_only:return
    cmd=[args.xemu,'-rom',str(args.rom),'-sdimg',str(image),'-8',str(disk),'-autoload','-besure']
    if args.fast:cmd.append('-sleepless')
    print('TinyBrain65 / V6.1 candidate: RETURN asks; F1 chooses an example; ESC clears or stops.',flush=True)
    subprocess.run(cmd,check=True)
if __name__=='__main__':
    try:main()
    except (OSError,AssertionError,RuntimeError,StopIteration) as e:
        print('TinyBrain65:',str(e) or 'Invalid private SD image.',file=sys.stderr);sys.exit(1)
