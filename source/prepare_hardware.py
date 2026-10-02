"""Focused manual facts, development paraphrases and a separately frozen test.

Keep V6's tokenizer, all old targets and all old training rows unchanged.
The user's screenshots are rehearsed regressions, never held-out evidence.
"""
import hashlib,json
from pathlib import Path
from tokenizer import ROOT,Tokenizer,REFUSAL,normalize

def save(name,value):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2))

def main():
    assert not (ROOT/'data/hardware-train.json').exists()
    knowledge=json.loads((ROOT/'data/knowledge.json').read_text());old={r['id']:r for r in knowledge}
    tok=Tokenizer.load();facts=[]
    def fact(key,answer,source,line,train,dev,test):
        if key in old:assert answer==old[key]['answer']
        facts.append(dict(id=key,answer=answer,source=source,line=line,train=train.split('|'),dev=dev.split('|'),test=test.split('|')))
    fact('sid-definition','sid is the sound interface device, used for sound and music.',
         'appendix-sid-registers.tex',1,
         'what is sid|what is the sid chip|what is a sid|what does sid stand for|explain sid|tell me about the sid|sid meaning|what does the sid do|is sid for sound|what is the sound interface device',
         'what job does sid have|sid means what exactly|please explain the sid sound chip',
         'sid is short for what|why does this computer have something called sid|tell me what sid actually is')
    fact('sound-chips','the mega65 has four sid sound chips.',
         'appendix-sid-registers.tex',5,
         'do you have a sound chip|what sound chip do you use|how many sids are there|which chip makes sound|what audio chip does the mega65 use|how many sound chips do you have|what is your sound hardware|does the mega65 have sound|do you have sid chips|what chips make music',
         'which hardware produces audio|does this machine contain a sid|how many sids does this computer have',
         'what is responsible for making noises in the mega65|name the audio chips inside this machine|count the built in sid chips')
    fact('sid-voices','each sid has three voices; four sids provide twelve voices.',
         'appendix-basic65.tex',7119,
         'how many voices does one sid have|how many voices do four sids have|how many sid voices are available|how many sound voices are there|sid voice count|can one sid play three voices',
         'how many voices per sid chip|what is the total sid voice count',
         'give me the number of voices in a single sid|with all four sids how many voices is that')
    fact('sid-vic','sid handles sound; vic-iv handles graphics.',
         'appendix-sid-registers.tex + appendix-viciv-registers.tex',1,
         'is sid the graphics chip|is vic-iv the sound chip|what is the difference between sid and vic|sid versus vic-iv|does sid draw graphics|does vic-iv make sound|which does sound sid or vic|is the sound chip called vic-iv',
         'does the sid handle video output|is vic the audio processor|sid and vic do the same thing right',
         'i thought sid drew the screen is that right|which one plays notes and which one draws pixels|would i use vic-iv for music or graphics')
    fact('music-capability','the mega65 can play music with sid chips; this helper only answers questions.',
         'appendix-sid-registers.tex + native/main.c',1,
         'can you make music|can you play music|can the mega65 make music|can this computer play music|do you make music|can you produce sound|can you generate a song|can tinybrain play a song|can this helper compose music',
         'are you able to make a tune|could this machine play a song|can this app create music for me',
         'will you perform a little tune for me|is making songs something tinybrain can do|is the mega65 capable of music')
    fact('sound-basic','use play for musical notes, sound for sound effects, and vol for volume.',
         'appendix-basic65.tex',7119,
         'how do i make music in basic|how do i make sound in basic|what commands make music|which basic commands control sound|how can i play notes|how do i make a sound effect|how do i control the volume|how do i use sound and music',
         'how can basic produce a tune|what basic words control audio|how can i make notes and effects',
         'point me to the basic commands for making noise|which commands handle notes effects and loudness|where do i start with music in basic')
    fact('display-modes','basic screen widths are 320 or 640 pixels; heights are 200 or 400.',
         'appendix-basic65.tex',8851,
         'what resolutions can you display|what screen resolutions do you support|what graphics resolutions are available|what are the basic screen sizes|which resolutions can basic use|how many pixels can the screen have|what resolution can you show|what display modes can basic use|what resolutions does the mega65 have',
         'what pixel sizes can i choose for basic graphics|which screen dimensions can basic use|what screen resolutions have you got',
         'list the width and height choices for a basic screen|how wide and tall can a basic bitmap be|tell me the available graphics sizes in basic')
    fact('pal-resolution','normal pal video output is 720x576 at 50 hz.',
         'appendix-viciv-registers.tex',188,
         'what is the pal output resolution|how many pixels in pal output|what resolution is pal|what is the pal video size|what is the pal frame rate|what refresh rate does pal use',
         'what size is the normal pal video signal|pal output has how many pixels',
         'give the usual pal output dimensions and refresh rate|what video timing does pal normally use')
    fact('ntsc-resolution','normal ntsc video output is 720x480 at 60 hz.',
         'appendix-viciv-registers.tex',192,
         'what is the ntsc output resolution|how many pixels in ntsc output|what resolution is ntsc|what is the ntsc video size|what is the ntsc frame rate|what refresh rate does ntsc use',
         'what size is the normal ntsc video signal|ntsc output has how many pixels',
         'give the usual ntsc output dimensions and refresh rate|what video timing does ntsc normally use')
    fact('terminal-resolution','this terminal uses a 640x200 display.',
         'native/m65_io.h',1,
         'what resolution does tinybrain use|what is this terminal resolution|what resolution does this app use|how big is the tinybrain display|what display size does this helper use',
         'how many pixels does your terminal use|what resolution is the tinybrain screen',
         'tell me the pixel dimensions of the tinybrain interface|what screen mode is this helper drawn in')
    reuse={
      'graphics-chip':('what is the graphics chip|which chip handles video|what chip draws the screen|do you have a graphics chip|what video chip do you use|what is vic-iv|what does the vic do|which chip makes graphics', 'which component produces the graphics|what hardware draws pictures', 'name the chip responsible for the display|tell me which device deals with video'),
      'cpu-name':('what is the cpu chip|which chip runs instructions|what processor do you have|what is the main processor|what is the 45gs02|which processor runs programs', 'which chip executes the code|what processor is inside you', 'name the unit that executes instructions|tell me the name of the mega65 processor'),
      'cpu-speed':('how fast is your processor|what is the cpu clock speed|how many mhz is the cpu|what clock rate do you run at', 'tell me the normal processor clock|what speed is the main cpu', 'how quickly is your processor clocked|what is the usual cpu frequency'),
      'total-memory':('how much ram do you have|how much memory does the machine have|what ram is built in|how much ram is there in total', 'tell me the installed memory sizes|what memory does a standard mega65 contain', 'how much chip and attic memory comes on the board|describe the standard ram capacity'),
      'cpu-window':('how much memory can the cpu see at once|what is the cpu address window|does the cpu see all ram at once', 'how large is the directly visible cpu memory window|what memory fits in the cpu window', 'explain the amount of memory visible to the cpu at one time|is all ram directly in the processor window'),
      'attic-vic':('can the graphics chip read attic ram|can vic-iv use attic memory|can video read attic ram', 'is attic memory directly available to the graphics chip|can i put visible graphics in attic memory', 'can the video controller fetch pixels directly from attic ram|does vic have direct access to attic memory')}
    for key,(tr,dv,te) in reuse.items():
        r=old[key];fact(key,r['answer'],r['source'],r.get('line',1),tr,dv,te)
    unknown_train=[
      'what song is playing now','what note is my sid currently playing','is my sid broken','what is my current screen resolution','what monitor am i using','what colour is the pixel on my screen right now','how loud are my speakers right now','which sid chip has failed','how much memory is free right now','what is my cpu temperature right now','is my sound cable plugged in','which game is making this noise','what resolution is my television','why does this unseen program make no sound','what are my current sid register values','what is the current volume','does the vic-v support 8k','what is the sid-x chip','how do i use the sidplay basic command','how do i use the sounddraw command','tell me the resolution of my phone','which speakers should i buy','compose a complete orchestral score','write a full sid player routine','why is the left speaker silent','is my monitor damaged','what is in memory address 50000 right now','which graphics mode is my game using']
    unknown_dev=['which tune is coming out of my speakers','how many bytes of ram are free now','what sid settings are active at this moment','what model of display is connected','tell me why my audio hardware is faulty','what does the basic command sidpaint do','what is the sid-v chip','what resolution is my laptop screen','why does the game i loaded have no music','what frequency is voice one playing now','what is my current video mode','is my speaker cable connected properly']
    unknown_test=['identify the track i can hear','read the sid filter register for me','which note is voice two playing at this instant','diagnose the buzzing noise from my speakers','name the monitor attached to this mega65','tell me the current width of my screen','how much ram did my last program leave free','is my right audio channel damaged','why did my picture disappear a minute ago','what does sidscreen do in basic','what does vicmusic do in basic','is the sid-z audio chip supported','what is the best graphics card for my pc','write the entire music player for my game','what is the brightness setting on my monitor','does the audio lead have a broken wire']
    train=[];dev=[];test=[];seen={}
    def row(key,q,a,split):
        q=normalize(q)
        if q in seen:
            assert seen[q]==(key,split),(q,seen[q],key,split);return
        seen[q]=(key,split)
        r=dict(id=key,question=q,answer=a,origin='hardware',group='scope' if key=='unknown' else 'hardware')
        p,c=tok.prompt(q),tok.completion(r)
        assert len(q)<=96 and len(p)<=80 and len(c)-2<=40 and len(p)+len(c)-1<=128,(q,a,len(p),len(c))
        {'train':train,'dev':dev,'test':test}[split].append(r)
    frames=['{}','{}?','please explain: {}','quick question: {}','{} please','im new here, {}','for my mega65, {}','hey, {}','i forgot, {}','just curious, {}','can you help, {}','{} thanks','for a demo, {}','on this computer, {}']
    for f in facts:
        for q in f['train']:
            for frame in frames:row(f['id'],frame.format(q),f['answer'],'train')
        for split in ['dev','test']:
            for q in f[split]:row(f['id'],q,f['answer'],split)
    for split,qs in [('train',unknown_train),('dev',unknown_dev),('test',unknown_test)]:
        for q in qs:
            for frame in (frames if split=='train' else ['{}']):row('unknown',frame.format(q),REFUSAL,split)
    old_train=json.loads((ROOT/'data/train.json').read_text());old_q={normalize(r['question']):r for r in old_train}
    for r in train:
        if r['question'] in old_q:assert old_q[r['question']]['answer']==r['answer'],r
    for split,rows in [('dev',dev),('test',test)]:
        rows[:]=[r for r in rows if r['question'] not in old_q]
    for name,rows in [('train',train),('dev',dev),('test',test)]:save('data/hardware-'+name+'.json',rows)
    # Add new fact records for native test helpers and the taught-questions list.
    for f in facts:
        if f['id'] not in old:knowledge.append({**f,'dev':[],'test':[]})
    save('data/knowledge.json',knowledge);save('data/hardware-facts.json',facts)
    observed=['can you make music','what is sid','what is a sid','what is the sid chip','do you have a sound chip','what resolutions can you display']
    save('data/hardware-user-regressions.json',[r for r in train if r['question'] in observed])
    report={'new_train':len(train),'new_dev':len(dev),'fresh_test':len(test),'facts':len(facts),'new_fact_count':len([f for f in facts if f['id'] not in old]),'parameters':375600,'tokenizer_unchanged':True,'test_sha256':hashlib.sha256((ROOT/'data/hardware-test.json').read_bytes()).hexdigest(),'note':'User examples are trained regressions. Dev is for selection; test is frozen before training and opened after selection. Manual facts are paraphrased; common BASIC sizes are not claimed to be all hardware modes.'}
    save('build/hardware-data.json',report);print(report)
if __name__=='__main__':main()
