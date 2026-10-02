"""Same-size V6 fine-tune; select with actual deployed integer inference.

The frozen new hardware test is never loaded here. V6's 699 regression
checks remain mandatory. Selection also limits losses on old development.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse,hashlib,json,random,time
import numpy as np
import torch
import torch.nn.functional as F
from tokenizer import ROOT,Tokenizer,PAD
from model import SidLM,enable_qat,plain_state
from export_model import export_checkpoint
from evaluate_pc import Core
from retention_eval import evaluate_rows
from train_pairs import score_report
from distill_loss import distillation_loss

def read(p):return json.loads((ROOT/p).read_text())
def save(p,v):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def evaluate(blob,out,baseline=None):
    core=Core(blob)
    reports={}
    for name,file in [('old-dev','data/dev.json'),('retention','data/retention-checks.json'),('hardware-dev','data/hardware-dev.json'),('screenshots','data/hardware-user-regressions.json')]:
        reports[name]=evaluate_rows(core,read(file))
    score_report(reports['old-dev'])
    old=reports['old-dev'];ret=reports['retention'];new=reports['hardware-dev'];user=reports['screenshots']
    losses=[]
    if baseline:
        losses=[dict(question=a['question'],before=a['actual'],after=b['actual']) for a,b in zip(baseline['results'],old['results']) if a['exact'] and not b['exact']]
    result=dict(model_sha256=sha(blob),retained=ret['exact'],retention_count=ret['count'],user_exact=user['exact'],user_count=user['count'],new_known=new['known_exact'],new_known_count=new['known_count'],new_refused=new['unknown_refused'],new_unknown_count=new['unknown_count'],old_known=old['known_exact'],old_refused=old['unknown_refused'],old_score=old['selection_score'],old_losses=len(losses),new_score=.7*new['known_exact']/new['known_count']+.3*new['unknown_refused']/new['unknown_count'])
    for name,r in reports.items():r['model_sha256']=result['model_sha256'];save(out/(name+'.json'),r)
    save(out/'old-losses.json',losses);save(out/'summary.json',result)
    return result,reports

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--new-rows',type=int,choices=[16,32],default=32);ap.add_argument('--steps',type=int,default=2500);ap.add_argument('--lr',type=float,default=.00001);ap.add_argument('--seed',type=int,default=6561);ap.add_argument('--preserve',type=float,default=0);args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);assert not (out/'history.json').exists()
    torch.set_num_threads(6);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    tok=Tokenizer.load();base=read('data/train.json');new=read('data/hardware-train.json');rows=base+new
    start=time.monotonic()
    fingerprint=hashlib.sha256((ROOT/'data/train.json').read_bytes()+(ROOT/'data/hardware-train.json').read_bytes()+(ROOT/'data/tokenizer.json').read_bytes()).hexdigest()
    cache=ROOT/'build'/('training-tensors-'+fingerprint[:12]+'.pt')
    if cache.exists():
        cached=torch.load(cache,weights_only=True);assert cached['sha256']==hashlib.sha256((ROOT/'data/train.json').read_bytes()+(ROOT/'data/hardware-train.json').read_bytes()+(ROOT/'data/tokenizer.json').read_bytes()).hexdigest()
    else:
        # The pre-existing V5 tensors cover the unchanged prefix of V6.
        previous=ROOT.parent/'megaqa-retention';old_file=previous/'data-v5/train.json'
        if old_file.exists() and (previous/'build/v5-training-tensors.pt').exists():
            old=json.loads(old_file.read_text())
            tensors=torch.load(previous/'build/v5-training-tensors.pt',weights_only=True)
            assert tensors['sha256']==hashlib.sha256(old_file.read_bytes()+(ROOT/'data/tokenizer.json').read_bytes()).hexdigest()
        else:
            # Portable packages can rebuild the same tensors without the old workspace.
            old=[];tensors={k:torch.empty((0,0),dtype=torch.float32 if k=='W' else torch.int64) for k in ['X','Y','W']}
        assert base[:len(old)]==old and len(tensors['X'])==len(old)
        encoded=[(tok.prompt(r['question']),tok.completion(r)) for r in rows[len(old):]]
        width=max(tensors['X'].shape[1],(max(len(p)+len(c)-1 for p,c in encoded)+7)//8*8)
        xs=[];ys=[];ws=[]
        for p,c in encoded:
            s=p+c;x=s[:-1];y=s[1:];w=[0.]*(len(p)-1)+[4.]+[1.]*(len(y)-len(p))
            xs.append(x+[PAD]*(width-len(x)));ys.append(y+[-100]*(width-len(y)));ws.append(w+[0.]*(width-len(w)))
        old_width=tensors['X'].shape[1]
        cached={key:torch.cat([F.pad(tensors[key],(0,width-old_width),value=pad),torch.tensor(vals)]) for key,vals,pad in [('X',xs,PAD),('Y',ys,-100),('W',ws,0)]}
        cached['sha256']=hashlib.sha256((ROOT/'data/train.json').read_bytes()+(ROOT/'data/hardware-train.json').read_bytes()+(ROOT/'data/tokenizer.json').read_bytes()).hexdigest();torch.save(cached,cache)
    X,Y,W=[cached[k].cuda() for k in ['X','Y','W']];width=X.shape[1];assert len(X)==len(rows)
    anchors=read('data/retention-indices.json');essential=[r['index'] for r in anchors if r['essential']];other=[r['index'] for r in anchors if not r['essential']]
    classes=[0 if r['id']=='unknown' else 1 if not r['id'].startswith(('command-','task-')) else 2 for r in base];counts=Counter(classes)
    weights=torch.tensor([[.2,.4,.4][k]/counts[k] for k in classes])
    new_groups={}
    for i,r in enumerate(new,len(base)):new_groups.setdefault(r['id'],[]).append(i)
    known_groups=[v for k,v in new_groups.items() if k!='unknown']
    rng=random.Random(args.seed+1);trng=torch.Generator().manual_seed(args.seed+2)
    resume=ROOT/'training/V6-start.pt';model=SidLM.load(resume).cuda();assert model.num_params()==375600;enable_qat(model)
    teacher=SidLM.load(resume).cuda().eval() if args.preserve else None
    if teacher:
        enable_qat(teacher)
        for p in teacher.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.03)
    calibration=[tok.prompt(r['question'])+tok.completion(r) for r in read('data/calibration.json')]
    baseline_path=ROOT/'trials/baseline-v6'
    if not (baseline_path/'summary.json').exists():
        baseline_path.mkdir(parents=True,exist_ok=True)
        export_checkpoint(resume,baseline_path/'MEGAQA.BIN',calibration)
        assert (baseline_path/'MEGAQA.BIN').read_bytes()==(ROOT/'build/V6.BIN').read_bytes(),'V6 re-export changed'
        evaluate(ROOT/'build/V6.BIN',baseline_path)
    baseline=read('trials/baseline-v6/old-dev.json');base_summary=read('trials/baseline-v6/summary.json')
    config={**vars(args),'out':str(out),'parameters':375600,'initial_sha256':sha(resume),'tensor_sha256':cached['sha256'],'fresh_test_sha256':sha(ROOT/'data/hardware-test.json'),'sampler':f'{args.new_rows} new rows (25% unknown, known balanced by fact); 48 retention (half essential); remaining V6 replay (.2 unknown/.4 curated/.4 command).','eligibility':'699/699 old retention; all 6 user screenshots; new-dev known >=80%, unknown >=80%; at most 8 lost old-dev successes; old-dev known >=baseline-8 and unknown >=baseline-2.','selection':'Eligible checkpoints ranked by new-dev .7 known + .3 unknown; then fewer old-dev losses, then old-dev score. Frozen hardware test never read.'}
    save(out/'experiment.json',config);print('START',config,flush=True);history=[]
    for step in range(1,args.steps+1):
        indices=rng.choices(essential,k=24)+rng.choices(other,k=24)
        indices+=rng.choices(new_groups['unknown'],k=args.new_rows//4)
        indices+=[rng.choice(rng.choice(known_groups)) for _ in range(args.new_rows*3//4)]
        indices+=torch.multinomial(weights,48-args.new_rows,replacement=True,generator=trng).tolist();assert len(indices)==96
        sel=torch.tensor(indices,device='cuda');model.train();opt.zero_grad(set_to_none=True)
        logits,_=model(X[sel]);raw=F.cross_entropy(logits.reshape(-1,1024),Y[sel].reshape(-1),ignore_index=-100,reduction='none').view(96,width)
        loss=(raw*W[sel]).sum()/W[sel].sum()
        if teacher:
            with torch.no_grad():teacher_logits,_=teacher(X[sel])
            old_weights=W[sel]*(sel<len(base))[:,None]
            kd,_=distillation_loss(logits,teacher_logits,Y[sel],old_weights)
            loss=loss+args.preserve*kd
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for g in opt.param_groups:g['lr']=args.lr*min(step/100,1)
        opt.step()
        if step%500==0 or step==args.steps:
            dest=out/f'step-{step:04d}';dest.mkdir();checkpoint=dest/'checkpoint.pt'
            torch.save({'config':asdict(model.cfg),'state':plain_state(model),'extra':{'step':step,'qat':True,'gate':True,'experiment':config}},checkpoint)
            export_checkpoint(checkpoint,dest/'MEGAQA.BIN',calibration)
            score,_=evaluate(dest/'MEGAQA.BIN',dest,baseline)
            eligible=score['retained']==score['retention_count']==699 and score['user_exact']==score['user_count']==6 and score['new_known']/score['new_known_count']>=.8 and score['new_refused']/score['new_unknown_count']>=.8 and score['old_losses']<=8 and score['old_known']>=base_summary['old_known']-8 and score['old_refused']>=base_summary['old_refused']-2
            score.update(step=step,eligible=eligible,seconds=time.monotonic()-start,loss=loss.item());save(dest/'summary.json',score)
            history.append(score);save(out/'history.json',history);print('CHECKPOINT',score,flush=True)
    print('DONE',round(time.monotonic()-start,1),'seconds',flush=True)
if __name__=='__main__':main()
