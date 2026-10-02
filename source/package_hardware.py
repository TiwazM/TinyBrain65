"""Publish a separate, fully labelled V6.1 trial with reproducible source."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from tokenizer import ROOT

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def main():
    out=ROOT.parents[1]/'outputs/TINYBRAIN65-V6.1-TRIAL';archive=out.parent/(out.name+'.zip')
    assert not out.exists() and not archive.exists()
    selection=read('build/hardware-selection.json');chosen=selection['selected'];sha=chosen['model_sha256']
    assert sha==digest(ROOT/'build/MEGAQA.BIN')
    assert chosen['retained']==699 and chosen['user_exact']==6
    assert read('build/pc-user.json')['exact']==17
    for name,h in read('build/runtime-unchanged.json').items():assert digest(ROOT/name)==h,name
    prg=digest(ROOT/'native/build/tinybr61.prg');disk=digest(ROOT/'build/TINYBR61.D81')
    for label,count in [('v61-normal',3),('v61-fast',9),('v61-folder',2)]:
        r=read('build/'+label+'-verification.json')
        assert len(r['results'])==count and r['model_sha256']==sha
        assert all(x['answer_matches_expected'] for x in r['results'])
        assert r['graphics']['composited_pixels_match'] and r['graphics']['background_preserved']
        assert r['graphics']['glow_load_status'] is None
        assert r['graphics']['glow_sha256']==digest(ROOT/'build/GLOW.BIN')
        if label=='v61-folder':assert r['folder'] and r['real_sd'] and r['prg_sha256']==digest(ROOT/'native/build/qa-folder.prg')
        else:assert r['production'] and r['disk'] and r['real_sd'] and r['prg_sha256']==prg and r['disk_sha256']==disk
        if label=='v61-normal':assert not r['fast']
    ready=read('build/v61-ready-graphics.json')
    assert ready['effects_loaded']==1 and ready['glow_load_status'] is None and ready['prg_sha256']==prg and ready['crt_preference_preserved']
    normal=read('build/v61-normal-verification.json')
    report=dict(passed=True,model_sha256=sha,prg_sha256=prg,disk_sha256=disk,parameters=375600,model_bytes=515638,inference_tokenizer_and_glow_unchanged=True,production_normal_questions=3,production_fast_questions=9,sd_folder_questions=2,normal_seconds=[round(r['wall_seconds'],2) for r in normal['results']],strict_replacement_gate_passed=selection['strict_replacement_gate_passed'],physical_mega65_tested=False)
    (ROOT/'build/v61-release-verification.json').write_text(json.dumps(report,indent=2))
    files=[]
    def copy(src,name):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p);files.append(p)
    def write(name,text):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf8');files.append(p)
    write('README.txt','''TINYBRAIN65 V6.1 / OPTIONAL HARDWARE TRAINING TRIAL
3 October 2026

Keep published V6. This is a separate candidate with more hardware facts,
not a regression-free replacement. The six recent sound/SID/resolution
examples are now correct, but unfamiliar wording can still produce nonsense.

RESULTS WITH THE ACTUAL INTEGER C RUNTIME
                                             V6       V6.1
Recent user examples (deliberately taught)      0/6       6/6
Protected historical cases (rehearsed)        699/699   699/699
Earlier user regressions (rehearsed)           17/17     17/17
New frozen hardware test: supported exact       7/38     20/38
New frozen hardware test: unknown refused      14/16     13/16
Older 144-question test: supported exact       52/80     60/80
Older 144-question test: unknown refused       48/64     46/64

These are exact matches against short taught answers, not general accuracy.
New questions were frozen before training and evaluated only after selecting
the candidate. The older test has historical exposure. The six user examples
and the 699 protected cases are training-informed, not independent evidence.

On the older test, 13 answers improve and 7 previously correct ones regress.
Two PEEK phrasings now get wrong answers; five formerly refused questions
now get unsupported answers. Full before/after results are in verification/.
Do not judge the candidate only by the six examples that prompted training.

The conservative replacement gate FAILED. Old development improves in
aggregate (893 to 993 supported exact answers), but 43 previous successes
are lost; only 8 were allowed. New hardware development also falls short
of the 80% supported target (28/38). The experimental candidate was selected
separately using development results, before opening the fresh test.

WHAT CHANGED
- Ten new hardware facts and broader wording for six existing facts.
- SID meaning, sound chips/voices, music capability, BASIC sound commands,
  SID versus VIC-IV, common BASIC dimensions, PAL/NTSC output, terminal size.
- Contrasts between CPU, graphics, sound, installed RAM and CPU memory window.
- More live-state and unsupported-device questions for abstention.
- V6 is used as a frozen training teacher on old examples to limit drift.
  That teacher is the same small model, and runs only on the training PC.

375,600 parameters; 4-bit weights; 515,638-byte model; unchanged tokenizer,
native inference core, context/answer limits and glow. No extra computation
per generated token was added. Answer length still determines total time.
There is no answer lookup for the new facts: they are generated by the model.
The pre-existing WarGames easter egg remains scripted and unchanged.

INSTALL ALONGSIDE V6
Copy SD-CARD/TINYBR61 to your SD card, keeping all three files together:

  CHDIR "/",U12
  CHDIR "TINYBR61",U12
  MOUNT "TINYBR61.D81"
  RUN "TINYBR61",U8

Use this trial's D81 AND MEGAQA.BIN together. Its model fingerprint differs
from V6, although the model filename stays MEGAQA.BIN. Keep your TINYBRG6
folder for the published build. GLOW.BIN is identical to the current V6 glow.
The startup glow success message remains removed. Badge: V6.1 * trust, but verify.

CONTROLS
RETURN asks; F1 cycles examples; DEL edits; ESC stops or clears the draft.
ESC with an empty draft clears the visible log. Each question starts fresh;
the visible conversation does not provide model memory of earlier turns.
Questions allow 96 characters / 80 tokens; answers stop at 40 tokens.

TRY THESE
  what is sid
  what is the sid chip
  do you have a sound chip
  can you make music
  what resolutions can you display
  what is my current screen resolution

The resolution answer describes common BASIC screen dimensions, not an
exhaustive list of hardware modes. The helper cannot inspect the live video
mode or play a song; MEGA65 hardware capability and app capability differ.

XEMU
RUN-XEMU.bat uses your installed Xemu/ROM and creates a PRIVATE copy of your
SD image (normally 4 GB). The original is unchanged. Normal speed is default;
RUN-XEMU.bat fast enables acceleration. Python 3 is needed for the launcher.
For alternate paths use: python tools/run_xemu.py --help
No emulator, ROM or SD image is included.

VERIFICATION
The production D81 passed three hardware questions at normal speed and nine
more checks accelerated, including old questions, refusal and the easter egg.
Native tokens, answers, streaming and actual composited pixels match the PC
reference. Two questions also pass after loading from the TINYBR61 SD folder.
CRT preservation and clean startup were checked. No physical MEGA65 test
is claimed for the candidate. Timings are in v61-release-verification.json.

TRAINING AND REPRODUCTION
Three runs start from the included V6 checkpoint. The first two use 1,862
new rows, 2,500 steps at learning rate .00001, with 16 or 32 new rows per
96-row batch. Both failed the replacement gate. The final run uses 10,832
broader rows, 4,000 steps at .00003, 32 new rows, 48 retention rows and
16 old replay rows. Seed 6561; unchanged quantization-aware training and
old export calibration. Supervised CE uses gate weight 4, answer 1; the
final run adds weight-1, temperature-2, gold-guarded KL on old examples.
Selected: expanded-preserve, step 4000. All candidate summaries are included.

The new 54-question test has 38 supported and 16 unsupported questions.
Hardware development has 38 supported and 12 unsupported questions. It was
used to revise training wording and select candidates; it is not a fresh test.
No further training or candidate selection was done after the test was opened.

source/ includes the final data, starting/selected checkpoints, runtime,
exporter and checks. The initial small dataset is kept separately. In source/:
  python train_hardware.py --out trials/reproduce --new-rows 32 --steps 4000 --lr .00003 --preserve 1
  python export_model.py --checkpoint training/best-qat.pt
  native\\build.bat
  python d81.py

Training needs CUDA PyTorch and NumPy. Native builds use LLVM-MOS; edit
tool paths for your machine. Floating-point/GPU differences can affect
retraining. The supplied exported model and its SHA256 identify this trial.
Training tensor caches can be rebuilt without the earlier workspace.

Manual-derived facts retain their source/line references. See SOURCE-NOTICES.txt,
GFDL-1.3.txt and source/sources/guide for documentation and licensing.
''')
    for name in ['TINYBR61.D81','MEGAQA.BIN','GLOW.BIN']:
        copy(ROOT/'build'/name,'SD-CARD/TINYBR61/'+name);copy(ROOT/'build'/name,'source/build/'+name)
    for name in ['run_xemu.py','fat_image.py']:copy(ROOT/name,'tools/'+name)
    write('RUN-XEMU.bat','@echo off\nsetlocal\nset "SPEED="\nif /i "%~1"=="fast" set "SPEED=--fast"\nwhere py >nul 2>nul\nif not errorlevel 1 (\n  py -3 "%~dp0tools\\run_xemu.py" %SPEED%\n) else (\n  python "%~dp0tools\\run_xemu.py" %SPEED%\n)\nif errorlevel 1 pause\n')
    for name in ['prepare_hardware.py','expand_hardware.py','train_hardware.py','distill_loss.py','select_hardware.py','prepare_runtime.py','package_hardware.py','model.py','quant.py','train.py','train_pairs.py','retention_eval.py','tokenizer.py','export_model.py','evaluate_pc.py','verify_xemu.py','monitor.py','d81.py','fat_image.py','run_xemu.py','make_title.py','make_glow.py','user-regressions.json','challenge_questions.json','benchmark-retention-frozen.json','xemu-hardware-normal.json','xemu-hardware-fast.json','xemu-hardware-folder.json','SOURCE-NOTICES.txt']:
        copy(ROOT/name,'source/'+name)
    for p in (ROOT/'native').iterdir():
        if p.suffix in ['.c','.h','.s','.ld','.bat']:copy(p,'source/native/'+p.name)
    for name in ['tinybr61.prg','tinybr61.map','qa-folder.prg','qa-folder.map','megaqa.dll']:copy(ROOT/'native/build'/name,'source/native/build/'+name)
    for p in (ROOT/'data').glob('*.json'):copy(p,'source/data/'+p.name)
    for p in (ROOT/'sources/guide').glob('*.tex'):copy(p,'source/sources/guide/'+p.name)
    copy(ROOT/'sources/GFDL-1.3.txt','GFDL-1.3.txt');copy(ROOT/'sources/GFDL-1.3.txt','source/sources/GFDL-1.3.txt');copy(ROOT/'SOURCE-NOTICES.txt','SOURCE-NOTICES.txt')
    for name in ['V6-start.pt','best-qat.pt']:copy(ROOT/'training'/name,'source/training/'+name)
    copy(ROOT/'build/V6.BIN','source/build/V6.BIN')
    for name in ['hardware-data','hardware-selection','hardware-comparison','runtime-unchanged','export-report','glow-asset','pc-user','v6-hardware-fresh','v61-hardware-fresh','v6-old-historical','v61-old-historical','v6-screenshots','v61-screenshots','v61-normal-verification','v61-fast-verification','v61-folder-verification','v61-ready-graphics','v61-release-verification']:
        copy(ROOT/'build'/(name+'.json'),'verification/'+name+'.json');copy(ROOT/'build'/(name+'.json'),'source/build/'+name+'.json')
    for trial in ['new16','new32','expanded-preserve']:
        for p in (ROOT/'trials'/trial).rglob('*.json'):copy(p,'source/'+p.relative_to(ROOT).as_posix())
    for p in (ROOT/'trials/baseline-v6').glob('*.json'):copy(p,'source/'+p.relative_to(ROOT).as_posix())
    copy(ROOT/'build/v61-ready.png','XEMU-READY.png');copy(ROOT/'build/v61-normal.png','XEMU-HARDWARE.png');copy(ROOT/'build/v61-fast.png','XEMU-HISTORY.png')
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files};write('SHA256.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(files):
            assert p.suffix.lower() not in ['.img','.rom','.pdf'];z.write(p,out.name+'/'+p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,m in manifest.items():
            data=z.read(out.name+'/'+name);assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256']
    print(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),files=len(files),manifest_verified=True))
if __name__=='__main__':main()
