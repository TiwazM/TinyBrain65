"""Same-size QAT fine-tune with paired examples and V4 replay; ordinary CE only."""
from dataclasses import asdict
from collections import Counter
from pathlib import Path
import argparse,hashlib,json,random,time
import numpy as np
import torch
import torch.nn.functional as F
from model import SidLM,enable_qat,plain_state
from tokenizer import ROOT,Tokenizer,PAD
from train import evaluate

def score_report(report):
    rows=report['results'];parts={}
    for name,selected in [
        ('old-known',[r for r in rows if r['dev_source']=='v4' and r['id']!='unknown']),
        ('old-unknown',[r for r in rows if r['dev_source']=='v4' and r['id']=='unknown']),
        ('new-command',[r for r in rows if r['dev_source']=='pairs' and r['id'].startswith('command-')]),
        ('new-decision',[r for r in rows if r['dev_source']=='pairs' and r['id']!='unknown' and not r['id'].startswith('command-')]),
        ('new-invented',[r for r in rows if r['dev_source']=='pairs' and r['id']=='unknown' and r['group']=='invented']),
        ('new-scope',[r for r in rows if r['dev_source']=='pairs' and r['id']=='unknown' and r['group']!='invented']),
    ]:
        assert selected,name
        parts[name]={'count':len(selected),'exact':sum(r['exact'] for r in selected),'refused':sum(r['abstained'] for r in selected)}
    rates={k:v['exact']/v['count'] for k,v in parts.items()}
    old=.65*rates['old-known']+.35*rates['old-unknown']
    new=.65*(rates['new-command']+rates['new-decision'])/2+.35*(rates['new-invented']+rates['new-scope'])/2
    report.update(selection_score=(old+new)/2,old_score=old,new_score=new,parts=parts)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['paired','replay'],default='paired');ap.add_argument('--seed',type=int,default=6512)
    ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--lr',type=float,default=.00005)
    ap.add_argument('--evaluate-only',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'history.json').exists()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    tok=Tokenizer.load();rows=json.loads((ROOT/'data/train.json').read_text());dev=json.loads((ROOT/'data/dev.json').read_text())
    resume=ROOT/'training/V4-start.pt';model=SidLM.load(resume).cuda();assert model.num_params()==375600
    enable_qat(model)
    if args.evaluate_only:
        report=score_report(evaluate(model,dev,tok));(args.out/'baseline-dev.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2));return
    classes=[];old_indices=[]
    for i,r in enumerate(rows):
        if r.get('origin')!='pairs':
            kind=0 if r['id']=='unknown' else 1 if not r['id'].startswith(('command-','task-')) else 2
            classes.append((r['origin'],kind));old_indices.append(i)
    counts=Counter(classes)
    weights=torch.tensor([.5*[.2,.4,.4][kind]/counts[(origin,kind)] for origin,kind in classes])
    old_indices=torch.tensor(old_indices);groups=json.loads((ROOT/'data/pair-groups.json').read_text())
    fingerprint=hashlib.sha256((ROOT/'data/train.json').read_bytes()+(ROOT/'data/tokenizer.json').read_bytes()).hexdigest()
    tensor_path=ROOT/'build'/('training-tensors-'+fingerprint[:12]+'.pt')
    if tensor_path.is_file():
        tensors=torch.load(tensor_path,weights_only=True);assert tensors['sha256']==fingerprint
    else:
        encoded=[(tok.prompt(r['question']),tok.completion(r)) for r in rows]
        width=(max(len(p)+len(c)-1 for p,c in encoded)+7)//8*8;xx=[];yy=[];ww=[]
        for p,c in encoded:
            s=p+c;x=s[:-1];y=s[1:];w=[0.]*(len(p)-1)+[4.]+[1.]*(len(y)-len(p))
            xx.append(x+[PAD]*(width-len(x)));yy.append(y+[-100]*(width-len(y)));ww.append(w+[0.]*(width-len(w)))
        tensors={'sha256':fingerprint,'X':torch.tensor(xx),'Y':torch.tensor(yy),'W':torch.tensor(ww)}
        torch.save(tensors,tensor_path)
    X=tensors['X'].cuda();Y=tensors['Y'].cuda();W=tensors['W'].cuda();width=X.shape[1]
    rng=torch.Generator().manual_seed(args.seed+9);pair_rng=random.Random(args.seed+10)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.03)
    config={**vars(args),'out':str(args.out),'initial_checkpoint_sha256':hashlib.sha256(resume.read_bytes()).hexdigest(),'parameters':model.num_params(),'train_sha256':hashlib.sha256((ROOT/'data/train.json').read_bytes()).hexdigest(),'dev_sha256':hashlib.sha256((ROOT/'data/dev.json').read_bytes()).hexdigest(),'fresh_benchmark_sha256':hashlib.sha256((ROOT/'benchmark-pairs-frozen.json').read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256((ROOT/'data/tokenizer.json').read_bytes()).hexdigest(),'batch_size':96,'sampling':'Paired mode: 48 V4 replay rows, 12 same-answer pairs, 6 different-intent pairs, 6 known/unknown scope pairs. Replay-only control: 96 V4 replay rows. V4 sampler is preserved within replay.','loss':'Ordinary teacher-forced cross-entropy: question weight 0, gate 4, answer 1. No contrastive loss or new head.','selection':'Half old development score, half new development score. Each is .65 known exact + .35 unknown refusal; new known splits command/curated equally and new unknown splits invented/live-scope equally. Fresh test is not scored during training.'}
    config['pair_groups_sha256']=hashlib.sha256((ROOT/'data/pair-groups.json').read_bytes()).hexdigest()
    (args.out/'experiment.json').write_text(json.dumps(config,indent=2));print(config,flush=True)
    best=-1;history=[];start=time.monotonic()
    for step in range(1,args.steps+1):
        indices=old_indices[torch.multinomial(weights,48 if args.mode=='paired' else 96,replacement=True,generator=rng)].tolist()
        if args.mode=='paired':
            for category,n in [('same',12),('decision',6),('scope',6)]:
                for _ in range(n):indices+=pair_rng.sample(pair_rng.choice(groups[category])['indices'],2)
        assert len(indices)==96
        sel=torch.tensor(indices,device='cuda');model.train();opt.zero_grad(set_to_none=True)
        logits,_=model(X[sel]);raw=F.cross_entropy(logits.reshape(-1,1024),Y[sel].reshape(-1),ignore_index=-100,reduction='none').view(96,width)
        loss=(raw*W[sel]).sum()/W[sel].sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in opt.param_groups:group['lr']=args.lr*min(step/100,1)
        opt.step()
        if step%500==0 or step==args.steps:
            report=score_report(evaluate(model,dev,tok));report.update(step=step,qat=True,seconds=time.monotonic()-start,loss=loss.item())
            (args.out/f'dev-{step}.json').write_text(json.dumps(report,indent=2))
            checkpoint={'config':asdict(model.cfg),'state':plain_state(model),'extra':{'step':step,'qat':True,'gate':True,'selection_score':report['selection_score'],'experiment':config}}
            torch.save(checkpoint,args.out/'latest.pt')
            if report['selection_score']>best:best=report['selection_score'];torch.save(checkpoint,args.out/'best-qat.pt')
            history.append({k:v for k,v in report.items() if k!='results'});(args.out/'history.json').write_text(json.dumps(history,indent=2))
            print('DEV',step,'score',round(report['selection_score'],4),'old',round(report['old_score'],4),'new',round(report['new_score'],4),'known',report['known_exact'],'/',report['known_count'],'unknown',report['unknown_refused'],'/',report['unknown_count'],'seconds',round(report['seconds'],1),flush=True)
    print('Finished',round(time.monotonic()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
