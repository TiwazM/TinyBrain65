"""Broaden semantic wording after development failures; keep the test frozen."""
import itertools,json,hashlib
from tokenizer import ROOT,Tokenizer,normalize,REFUSAL

def main():
    p=ROOT/'data/hardware-train.json';previous=p.read_bytes()
    (ROOT/'data/hardware-train-initial.json').write_bytes(previous)
    rows=json.loads(previous);facts={r['id']:r for r in json.loads((ROOT/'data/hardware-facts.json').read_text())}
    old={r['question']:r for r in json.loads((ROOT/'data/train.json').read_text())}
    # Only question strings are used for overlap exclusion, never test answers.
    excluded={r['question'] for name in ['dev','test'] for r in json.loads((ROOT/f'data/hardware-{name}.json').read_text())}
    seen={r['question']:r['id'] for r in rows};tok=Tokenizer.load()
    def add(key,q):
        q=normalize(q);a=REFUSAL if key=='unknown' else facts[key]['answer']
        if q in seen or q in excluded or q in old:return
        row=dict(id=key,question=q,answer=a,origin='hardware-expanded',group='scope' if key=='unknown' else 'hardware')
        pr,co=tok.prompt(q),tok.completion(row)
        if len(q)>96 or len(pr)>80 or len(pr)+len(co)-1>128:return
        rows.append(row);seen[q]=key
    for role,adjectives in [('sound-chips',['sound','audio','music']),('graphics-chip',['graphics','video','display']),('cpu-name',['cpu','central processing','main processing'])]:
        for adj,thing in itertools.product(adjectives,['chip','hardware','processor','device','unit']):
            noun=f'{adj} {thing}'
            for frame in ['what is your {}','which {} do you have','name the {}','what {} is in the mega65','tell me about the {}','what is the name of the {}','identify the {}','which {} is built in','what {} does this computer use','does this machine have a {}']:
                add(role,frame.format(noun))
    for role,functions in [('sound-chips',['makes sound','produces audio','generates music','plays notes','creates sound','handles audio output']),('graphics-chip',['draws pictures','produces graphics','draws the screen','generates video','handles video output','renders graphics']),('cpu-name',['runs code','executes instructions','executes programs','does the calculations','runs the software'])]:
        for noun,fn,frame in itertools.product(['chip','component','hardware','processor','part'],functions,['which {} {}','what {} {}','tell me which {} {}','i want to know what {} {}']):add(role,frame.format(noun,fn))
    for noun in ['sid','the sid','the sid chip','sid sound chip']:
        for frame in ['what job does {} do','what is the role of {}','what does {} mean','explain what {} is for','what is {} short for','expand the name {}','what is the purpose of {}','{} means what','what does the name {} stand for','i want to understand {}','describe {} to me','what does {} actually do']:
            add('sid-definition',frame.format(noun))
    for noun in ['sids','sid chips','sound chips','sid devices']:
        for frame in ['how many {} are built in','how many {} does this machine contain','count the {}','tell me the number of {}','how many {} do i get','how many {} are inside the mega65']:
            add('sound-chips',frame.format(noun))
    for noun in ['one sid','each sid','a single sid chip','all four sids','the four sound chips']:
        for frame in ['how many voices in {}','what is the voice count for {}','{} has how many voices','how many voices does {} provide','tell me about voices on {}']:
            add('sid-voices',frame.format(noun))
    for sound,video in itertools.product(['sid','the sid chip'],['vic','vic-iv','the video chip']):
        for frame in ['compare {} with {}','what does {} do compared to {}','{} and {} have what roles','is {} the same as {}','what are {} and {} used for']:
            add('sid-vic',frame.format(sound,video))
    for noun in ['sid','the sid chip']:
        for frame in ['does {} handle video','is {} used for graphics','does {} draw the screen','can {} render pixels','i thought {} was the video chip']:
            add('sid-vic',frame.format(noun))
    for noun in ['vic','vic-iv','the vic chip']:
        for frame in ['does {} handle audio','is {} used for music','does {} play notes','can {} generate sound','i thought {} was the sound chip']:
            add('sid-vic',frame.format(noun))
    for subject,verb,obj in itertools.product(['you','the mega65','this machine','this helper','tinybrain','this app'],['make','play','produce','create','generate','compose'],['music','a song','a tune','musical notes']):
        for frame in ['can {} {} {}','could {} {} {}','is it possible for {} to {} {}','are {} able to {} {}']:
            add('music-capability',frame.format(subject,verb,obj))
    for action in ['make music','produce a tune','make noise','play notes','make sound effects','control audio','adjust loudness','produce sound']:
        for frame in ['how can i {} in basic','how do i {} using basic','what basic commands {}','which commands let me {}','how would i use basic to {}','where do i start to {} in basic']:
            add('sound-basic',frame.format(action))
    for noun in ['screen sizes','graphics sizes','pixel dimensions','display sizes','screen resolutions','bitmap dimensions','graphics resolutions','screen widths and heights','pixel sizes']:
        for frame in ['what {} can basic use','which {} are available in basic','what {} do you support','which {} can i choose','tell me the available {}','list the possible {}','what {} does this machine have','what {} have you got','explain the {} in basic','which {} are possible with screen']:
            add('display-modes',frame.format(noun))
    for std in ['pal','ntsc']:
        for noun in ['resolution','output size','pixel dimensions','frame rate','video timing','refresh rate']:
            for frame in ['what {} does '+std+' use','tell me the '+std+' {}','give the normal '+std+' {}','what are the {} for '+std,'what is the usual {} in '+std]:add(std+'-resolution',frame.format(noun))
        for frame in ['how wide and tall is {} output','{} video has how many pixels','how many pixels are in the {} signal','what size is the {} frame','describe the normal {} video signal']:
            add(std+'-resolution',frame.format(std))
    for noun in ['tinybrain','this app','this terminal','the helper interface']:
        for frame in ['what pixel dimensions does {} use','how wide and tall is {}','what is the display size of {}','how many pixels in {}','what screen mode does {} use']:
            add('terminal-resolution',frame.format(noun))
    for device in ['sid-y','sid-q','sid-200','vic-vi','vic-x']:
        for frame in ['what is {}','what does {} do','do you have a {} chip','how many voices does {} have','what resolution can {} display']:add('unknown',frame.format(device))
    for thing in ['sid register','volume','screen resolution','video mode','free memory','cpu temperature','playing note','audio frequency','monitor model']:
        for frame in ['what is my current {}','tell me my present {}','can you read the current {}','what {} is active right now','show the live {}']:
            add('unknown',frame.format(thing))
    # Vary politeness only after adding genuinely different semantic forms.
    added=rows[len(json.loads(previous)):].copy()
    for r in added:
        for frame in ['{}?','please, {}','hey {}','for my machine, {}']:
            add(r['id'],frame.format(r['question']))
    p.write_text(json.dumps(rows,indent=2))
    print({'initial':len(json.loads(previous)),'expanded':len(rows),'test_sha256':hashlib.sha256((ROOT/'data/hardware-test.json').read_bytes()).hexdigest()})
if __name__=='__main__':main()
