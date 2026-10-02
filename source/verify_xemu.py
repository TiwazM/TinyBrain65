"""Own an Xemu process; compare native tokens and observe live streaming."""
import argparse,hashlib,json,re,socket,subprocess,time
from pathlib import Path
from monitor import Monitor as BaseMonitor
from evaluate_pc import Core
from tokenizer import ROOT

class Monitor(BaseMonitor):
    def memory(self,addr,n):
        out=bytearray()
        for a in range(addr,addr+n,256):
            rows=re.findall(r':[0-9a-fA-F]{8}:([0-9a-fA-F]{32})',self.command(f'M{a:x}'))
            assert len(rows)==16;out.extend(bytes.fromhex(''.join(rows)))
        return bytes(out[:n])
    def put(self,addr,data):return self.command('s%x '%addr+' '.join(f'{b:02x}' for b in data))
    def screen(self):
        data=self.memory(0x12000,2000)
        def ch(v):return chr(v+96) if 1<=v<=26 else '|' if v==125 else '-' if v==126 else '_' if v in (100,127) else '@' if v==0 else chr(v) if 32<=v<125 else '#'
        return '\n'.join(''.join(ch(v) for v in data[i:i+80]).rstrip() for i in range(0,2000,80))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fast',action='store_true');ap.add_argument('--disk',action='store_true');ap.add_argument('--real-sd',action='store_true');ap.add_argument('--release',action='store_true');ap.add_argument('--folder',action='store_true');ap.add_argument('--bad-model',choices=['missing','corrupt']);ap.add_argument('--bad-effects',choices=['missing','truncated']);ap.add_argument('--label');ap.add_argument('--questions',type=Path);ap.add_argument('--ready-only',action='store_true');ap.add_argument('--crt',choices=['on','off']);ap.add_argument('--video',choices=['pal','ntsc'],default='pal');args=ap.parse_args()
    assert not args.disk or args.release,'The disk autoloads the production program.'
    assert not args.folder or (args.real_sd and not args.release and not args.disk)
    name='tinybr61' if args.release else 'qa-folder' if args.folder else 'qa-test';prg=ROOT/'native/build'/f'{name}.prg';label=args.label or ('fast' if args.fast else 'normal')
    symbols={m[4]:int(m[1],16) for m in re.finditer(r'^\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+\d+\s+(\w+)\s*$',prg.with_suffix('.map').read_text(),re.M)}
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    cmd=[r'C:\Program Files\xemu\xmega65.exe','-headless','-nosound','-besure','-rom',r'C:\Users\lostl\AppData\Roaming\xemu-lgb\mega65\MEGA65.ROM','-sdimg',str(ROOT/'test-sd.img'),'-uartmon',f':{port}','-screenshot',str(ROOT/'build'/f'{label}.png')]
    model_dir=ROOT/'build'
    if args.bad_model:
        model_dir=ROOT/'build'/args.bad_model;model_dir.mkdir(exist_ok=True)
        if args.bad_model=='corrupt':
            damaged=bytearray((ROOT/'build/MEGAQA.BIN').read_bytes());damaged[1000]^=1
            (model_dir/'MEGAQA.BIN').write_bytes(damaged)
    if args.bad_effects:
        assert not args.real_sd and not args.bad_model
        model_dir=ROOT/'build'/('glow-'+args.bad_effects);model_dir.mkdir(exist_ok=True)
        (model_dir/'MEGAQA.BIN').write_bytes((ROOT/'build/MEGAQA.BIN').read_bytes())
        if args.bad_effects=='truncated':(model_dir/'GLOW.BIN').write_bytes((ROOT/'build/GLOW.BIN').read_bytes()[:1000])
    if not args.real_sd:cmd+=['-hdosvirt','-hdosdir',str(model_dir)]
    if args.disk:cmd+=['-8',str(ROOT/'build/TINYBR61.D81'),'-autoload']
    else:cmd+=['-prg',str(prg)]
    if args.fast:cmd+=['-sleepless']
    cmd+=['-videostd','1' if args.video=='ntsc' else '0','-lockvideostd','-allowscanlines']
    si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
    log=(ROOT/'build'/f'{label}-xemu.log').open('w')
    p=subprocess.Popen(cmd,startupinfo=si,stdout=log,stderr=log);m=None;core=Core();results=[]
    def value(name,n=1):return int.from_bytes(m.memory(symbols[name],n),'little')
    def check_text_layer():
        # Check actual displayed pixels, not only the debug text mirror.
        # Freeze at a completed turn so the cursor cannot change mid-capture.
        m.command('t1')
        try:
            logical=m.memory(0x12000,2000);colours=m.memory(0x12800,2000)
            video=m.memory(0x14000,4000);attrs=m.memory(0xff80000,4000)
            pixels=m.memory(0x40000,128000);font=m.memory(0x16800,2048)
            cursor=value('cursor_row')*80+value('cursor_col')
        finally:m.command('t0')
        assert attrs==bytes(4000),'Unexpected per-tile attributes / GOTOX'
        for cell in range(2000):
            assert int.from_bytes(video[cell*2:cell*2+2],'little')==4096+cell
            if cell==cursor:continue # May be paused halfway through one cursor update.
            off=cell*64;expected=bytearray(art[off:off+64]);glyph=logical[cell]
            for y,bits in enumerate(font[glyph*8:glyph*8+8]):
                for x in range(8):
                    if bits&(128>>x):expected[y*8+x]=colours[cell]
            assert pixels[off:off+64]==expected,('composited pixels',cell//80,cell%80)
    def wait_ready(limit=120):
        end=time.monotonic()+limit
        while time.monotonic()<end:
            v=value('status')
            if v==3:return
            if v==255:raise RuntimeError(m.screen())
            time.sleep(.1)
        raise TimeoutError(m.screen())
    def key(ch):
        if args.release:
            # Real hardware virtual-key registers, documented in the guide's
            # appendix-keyboard.tex. No test hook in the production executable.
            ascii_key=ch if isinstance(ch,int) else ord(ch)
            plain={20:0,8:0,13:1,241:4,27:71,32:60,ord('?'):55}
            for scan,chars in [(8,'3wa4zs'),(16,'5rd6cftx'),(24,'7yg8bhuv'),(32,'9ij0mkon'),(40,'+pl-.:@,'),(48,'*;'),(56,'1')]:
                if scan==48:
                    plain[ord('*')]=49;plain[ord(';')]=50
                else:
                    for i,c in enumerate(chars):plain[ord(c)]=scan+i
            plain.update({ord('e'):14,ord('q'):62,ord('2'):59,ord('/'):55,ord('='):53})
            scan=plain[ascii_key]
            if ascii_key==ord('?'):m.put(0xffd3616,b'\x0f')
            m.put(0xffd3615,bytes([scan]));time.sleep(.04)
            m.put(0xffd3615,b'\x7f')
            if ascii_key==ord('?'):m.put(0xffd3616,b'\x7f')
            time.sleep(.04);return
        m.put(symbols['test_key'],bytes([ch]) if isinstance(ch,int) else ch.encode())
        end=time.monotonic()+4
        while value('test_key'):
            if time.monotonic()>end:raise TimeoutError('test key not consumed')
            time.sleep(.01)
    try:
        end=time.monotonic()+80
        while time.monotonic()<end:
            if p.poll() is not None:raise RuntimeError(('Xemu exited',p.returncode))
            try:m=Monitor(port);break
            except OSError:time.sleep(.2)
        assert m
        if args.bad_model:
            end=time.monotonic()+120
            while time.monotonic()<end and value('status')!=255:time.sleep(.1)
            assert value('status')==255,m.screen()
            assert 'missing or damaged' in m.screen()
            (ROOT/'build'/f'{label}-verification.json').write_text(json.dumps({'model_error':args.bad_model,'rejected':True,'production':args.release,'prg_sha256':hashlib.sha256(prg.read_bytes()).hexdigest()},indent=2))
            print('PASS rejected',args.bad_model,'model',flush=True);return
        wait_ready();controls={}
        if args.crt:
            # Restart the unmodified PRG at its real entry with the user's
            # video effects enabled/disabled BEFORE any startup instructions.
            entry=int(re.search(r'^\s*([0-9a-f]+).*\s_start =',prg.with_suffix('.map').read_text(),re.M)[1],16)
            m.command('t1');old=m.memory(0xffd3053,2)
            crt=0x28 if args.crt=='on' else 0
            m.put(0xffd3053,bytes([(old[0]&~0x40)|(0x40 if args.crt=='on' else 0),(old[1]&~0x28)|crt]))
            m.put(symbols['status'],b'\0');m.command(f'g{entry:x}');m.command('t0');wait_ready()
            assert value('video_before_54')&0x28==crt,'CRT setup did not reach program entry'
            # Xemu does not retain/emulate SHDEMU in D053 across rasters.
            # Production never writes D053; only PALEMU/SMTH are asserted here.
        art=bytes([33])*128000 if args.bad_effects else (ROOT/'build/GLOW.BIN').read_bytes()
        loaded=m.memory(0x8710000,len(art))
        if loaded!=art:(ROOT/'build'/f'{label}-backdrop.bin').write_bytes(loaded)
        graphics={'effects_loaded':value('effects_loaded'),'glow_sha256':hashlib.sha256(loaded).hexdigest(),
                  'vic_registers':m.memory(0xffd3000,128).hex(),
                  'first_video_row':m.memory(0x14000,160).hex(),
                  'first_colour_row':m.memory(0xff80000,160).hex(),
                  'video_standard':args.video,'crt_test':args.crt,'prg_sha256':hashlib.sha256(prg.read_bytes()).hexdigest(),
                  'video_before_54':value('video_before_54'),'video_after_54':value('video_after_54'),
                  'video_before_53':value('video_before_53'),'video_after_53':value('video_after_53')}
        (ROOT/'build'/f'{label}-graphics.json').write_text(json.dumps(graphics,indent=2))
        assert graphics['effects_loaded']==(0 if args.bad_effects else 1)
        expected_status='GLOW.BIN missing or incomplete; glow off.' if args.bad_effects else None
        if expected_status:assert expected_status in m.screen(),'Missing artwork failure status'
        else:assert 'glow loaded.' not in m.screen(),'Unexpected artwork success message'
        graphics['glow_load_status']=expected_status
        assert loaded==art,'Background memory does not match artwork'
        check_text_layer();graphics['composited_pixels_match']=True
        regs=bytes.fromhex(graphics['vic_registers'])
        assert regs[0x5e]==80 and int.from_bytes(regs[0x58:0x5a],'little')==160
        assert (graphics['video_before_54']^graphics['video_after_54'])&0x68==0
        assert (graphics['video_before_53']^graphics['video_after_53'])&0x40==0
        graphics['crt_preference_preserved']=True;graphics['single_layer']=True
        graphics['fallback_verified']=bool(args.bad_effects)
        (ROOT/'build'/f'{label}-graphics.json').write_text(json.dumps(graphics,indent=2))
        fixed_header=m.memory(0x12000,8*80);fixed_footer=m.memory(0x12000+23*80,2*80)
        # Observe the real native block cursor in both phases, before typing.
        phases=set();end=time.monotonic()+2
        while time.monotonic()<end:
            phases.add(m.memory(0x12000+20*80+3,1)[0])
            if phases=={32,127}:break
            time.sleep(.03)
        assert phases=={32,127},('Blinking block cursor',phases)
        controls['blinking_block_cursor']=True
        if args.ready_only:
            screen=m.screen();(ROOT/'build'/f'{label}-screen.txt').write_text(screen)
            # Capture the visible phase without changing the program's UI state.
            end=time.monotonic()+2;saw_off=False
            while time.monotonic()<end:
                on=value('cursor_on');saw_off|=not on
                if saw_off and on:time.sleep(.08);break
                time.sleep(.02)
            print(screen,flush=True);return
        if args.release:
            key(241);assert m.memory(symbols['question'],97).split(b'\0')[0]==b'what are you';controls['hardware_f1']=True
        if args.fast and not args.release:
            key(241);assert m.memory(symbols['question'],97).split(b'\0')[0]==b'what are you';controls['f1_example']=True
            for _ in range(12):key(241)
            assert m.memory(symbols['question'],97).split(b'\0')[0]==b'what are you';controls['twelve_examples_wrap']=True
            key(27);key(13);assert 'Type a question' in m.screen();controls['empty_question']=True
            m.put(symbols['question'],b'~'*95+b'\0');key('~');key('x')
            assert m.memory(symbols['question'],97)==b'~'*96+b'\0';controls['input_length_limit']=True
            key(13);assert 'Please use a shorter question.' in m.screen();controls['token_limit']=True
            key(27);key('X');key(20);assert m.memory(symbols['question'],1)==b'\0';controls['case_and_delete']=True
            m.put(symbols['question'],b'what does edma do?\0');key(13)
            end=time.monotonic()+30
            while value('generated_count',2)<1 and time.monotonic()<end:time.sleep(.02)
            key(27);wait_ready();assert 'stopped ' in m.screen();controls['escape_during_answer']=True
        questions=['what does edma do?','how much attic ram is there?','how do i go to the sd root?']
        if args.fast:questions+=['what is the difference between peek and poke?','what cpu does the mega65 use?','can vic-iv read attic ram?','what is the weather today?','how do i make the border black?','how much memory do you have','what is your cpu speed','what are you','how do i run a game','what is the temperature of the cpu']
        if args.release and not args.fast:questions=['how much memory do you have','what is your cpu speed','what are you','how do i run a game','how do i load a program','what is the temperature of the cpu','how do i list files']
        if args.fast:questions+=['how do i start a prg','how do i start a d81','how do i mount dload run','what do you use for graphics']
        if args.release and not args.fast:questions=['what are you','how do i start a prg','how do i start a d81','how do i mount dload run','what do you use for graphics','how do i list files']
        if args.questions:questions=json.loads(args.questions.read_text())
        mixed_suite=any(isinstance(q,dict) for q in questions)
        for index,item in enumerate(questions):
            spec=item if isinstance(item,dict) else {'question':item};q=spec['question']
            scripted='scripted_answer' in spec
            expected={'actual':spec['scripted_answer'],'ids':[],'prompt_ids':core.tokenize(q),'abstained':False,'gate_token':0} if scripted else core.answer(q)
            if spec.get('clear_history'):key(27)
            if index==0 or args.release:
                if m.memory(symbols['question'],1)!=b'\0':key(27)
                for c in q:key(c)
            else:m.put(symbols['question'],q.encode()+b'X\0');key(8)
            previous_turns=value('history_turns',2)
            m.put(symbols['generated_count'],b'\0\0');key(13);start=time.monotonic();events=[];last=-1
            end=start+180
            while time.monotonic()<end:
                state=value('status');count=value('generated_count',2)
                if count!=last:
                    events.append({'wall_seconds':time.monotonic()-start,'state':state,'tokens':count});last=count
                if state==3 and value('history_turns',2)>previous_turns:break
                if state==255:raise RuntimeError(m.screen())
                time.sleep(.02 if args.fast else .12)
            else:raise TimeoutError(m.screen())
            wall_seconds=time.monotonic()-start
            actual=m.memory(symbols['output_text'],384).split(b'\0',1)[0].decode('ascii')
            check_text_layer()
            ids=list(int.from_bytes(m.memory(symbols['generated_ids']+i*2,2),'little') for i in range(count))
            assert actual==expected['actual'],(q,actual,expected['actual'])
            assert ids==expected['ids'];n=value('prompt_count',2)
            assert value('abstained')==int(expected['abstained'])
            assert value('gate_token')==expected['gate_token']
            assert value('scripted_reply')==int(scripted),(q,'Wrong model/script dispatch')
            prompt=[int.from_bytes(x,'little') for x in [m.memory(symbols['prompt_ids']+i*2,2) for i in range(n)]]
            assert prompt==expected['prompt_ids']
            record={'question':q,'answer':actual,'abstained':expected['abstained'],'gate_token':expected['gate_token'],'prompt_tokens':n,'answer_tokens':count,'seconds':value('elapsed_seconds',2),'wall_seconds':wall_seconds,'first_token_seconds':value('first_token_seconds',2),'streaming_events':events,'scripted_reply':scripted,'answer_matches_expected':True,'native_tokens_exact':not scripted}
            if count>1:assert any(e['state']==2 and 0<e['tokens']<count for e in events),'No streaming observed'
            if not args.fast:
                assert abs(record['seconds']-wall_seconds)<2.5,record
                first_wall=wall_seconds if expected['abstained'] or scripted else next(e['wall_seconds'] for e in events if e['tokens']>0)
                assert abs(record['first_token_seconds']-first_wall)<2.5
            results.append(record);print('PASS',q,'=>',actual,record['seconds'],'s; first',record['first_token_seconds'],'s',flush=True)
            assert m.memory(symbols['question'],1)==b'\0','Composer did not reset'
            assert m.memory(0x12000,8*80)==fixed_header and m.memory(0x12000+23*80,2*80)==fixed_footer
            visible=m.screen();record['screen']=visible;record['history_scrolls']=value('history_scrolls',2)
            assert 'waiting for question' in visible
            assert actual in ' '.join(visible.split()),'Latest answer was clipped or wrapping changed its text'
            rows=visible.split('\n')[8:19]
            for i,row in enumerate(rows):
                if i and row.strip().startswith('you>'):assert not rows[i-1].strip(),'Missing blank line between turns'
            if len(results)==2 and not mixed_suite:
                assert results[0]['answer'] in ' '.join(visible.split()),'Previous answer disappeared too early'
        screen=m.screen();(ROOT/'build'/f'{label}-screen.txt').write_text(screen)
        report={'fast':args.fast,'disk':args.disk,'real_sd':args.real_sd,'folder':args.folder,'production':args.release,'controls':controls,'prg_sha256':hashlib.sha256(prg.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(core.blob).hexdigest(),'results':results}
        if args.disk:report['disk_sha256']=hashlib.sha256((ROOT/'build/TINYBR61.D81').read_bytes()).hexdigest()
        report['vic_registers']=m.memory(0xffd3000,128).hex()
        report['graphics']=graphics
        assert m.memory(0x8710000,len(art))==art,'Scrolling modified the background cache'
        report['graphics']['background_preserved']=True
        report['history_turns']=value('history_turns',2);report['history_scrolls']=value('history_scrolls',2)
        report['fixed_header_footer']=True
        report['blank_line_between_turns']=True
        if len(results)>=9 and not mixed_suite:
            assert report['history_scrolls']>0 and results[0]['answer'] not in ' '.join(screen.split())
            report['oldest_answer_scrolled_off']=True
        (ROOT/'build'/f'{label}-verification.json').write_text(json.dumps(report,indent=2));print(screen,flush=True)
        time.sleep(.5)  # Give Xemu a full rendered frame before its exit capture.
    finally:
        if m:
            try:m.command('~exit')
            except (OSError,RuntimeError):pass
            m.sock.close()
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
        log.close()
if __name__=='__main__':main()
