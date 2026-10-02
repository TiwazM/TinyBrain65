"""Train a real autoregressive transformer on the reviewed narrow Q/A corpus."""
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
import argparse,json,random,time,hashlib,copy
import numpy as np
import torch
from model import SidLM,ModelConfig,enable_qat,plain_state
from tokenizer import Tokenizer,PAD,EOS,KNOWN,UNKNOWN,FIRST_TEXT,REFUSAL,ROOT

@torch.inference_mode()
def evaluate(model,items,tok):
    model.eval();groups=defaultdict(list);results=[None]*len(items)
    for n,item in enumerate(items):groups[len(tok.prompt(item['question']))].append(n)
    for group in groups.values():
        for start in range(0,len(group),64):
            ns=group[start:start+64];x=torch.tensor([tok.prompt(items[n]['question']) for n in ns],device='cuda');cache=model.new_cache(len(ns))
            generated=[[] for _ in ns];done=[False]*len(ns);abstained=[False]*len(ns)
            for step in range(41):
                logits,_=model(x,cache=cache)
                if step==0:nxt=logits[:,-1,KNOWN:UNKNOWN+1].argmax(-1)+KNOWN
                else:
                    score=logits[:,-1].clone();score[:,:EOS]=-float('inf');score[:,KNOWN:FIRST_TEXT]=-float('inf');nxt=score.argmax(-1)
                for k,v in enumerate(nxt.tolist()):
                    if not done[k]:
                        if step==0:
                            abstained[k]=v==UNKNOWN;done[k]=abstained[k]
                        elif v==EOS:done[k]=True
                        else:generated[k].append(v)
                if all(done):break
                x=nxt[:,None]
            for n,g,ended,refused in zip(ns,generated,done,abstained):
                answer=REFUSAL if refused else tok.decode(g);unknown=items[n]['id']=='unknown'
                results[n]={**items[n],'actual':answer,'abstained':refused,'exact':refused if unknown else answer==items[n]['answer'],'ended':ended}
    known=[r for r in results if r['id']!='unknown'];unknown=[r for r in results if r['id']=='unknown']
    k=sum(r['exact'] for r in known);u=sum(r['abstained'] for r in unknown)
    score=.65*k/max(1,len(known))+.35*u/max(1,len(unknown))
    return {'count':len(results),'exact':sum(r['exact'] for r in results),'accuracy':sum(r['exact'] for r in results)/len(results),'selection_score':score,'known_exact':k,'known_count':len(known),'known_refused':sum(r['abstained'] for r in known),'unknown_refused':u,'unknown_count':len(unknown),'results':results}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=6000);ap.add_argument('--resume');ap.add_argument('--qat-start',type=int,default=4000);args=ap.parse_args()
    torch.manual_seed(6502);random.seed(6502);np.random.seed(6502);torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    tok=Tokenizer.load();items=json.loads((ROOT/'data/train.json').read_text());dev=json.loads((ROOT/'data/dev.json').read_text())
    cfg=ModelConfig(vocab=len(tok.tokens),n_ctx=128,d_model=96,n_layer=3,n_head=4,d_ff=192)
    model=SidLM.load(args.resume) if args.resume else SidLM(cfg);model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.03)
    xx=[];yy=[];masks=[];kinds=[]
    max_len=max(len(tok.prompt(i['question']))+len(tok.completion(i))-1 for i in items)
    width=(max_len+7)//8*8;assert width<=128
    for item in items:
        p=tok.prompt(item['question']);s=p+tok.completion(item);x=s[:-1];y=[-100]*(len(p)-1)+s[len(p):]
        mask=[0.]*(len(p)-1)+[4.]+[1.]*(len(y)-len(p))
        xx.append(x+[PAD]*(width-len(x)));yy.append(y+[-100]*(width-len(y)));masks.append(mask+[0.]*(width-len(mask)))
        kinds.append(0 if item['id']=='unknown' else 1 if not item['id'].startswith(('command-','task-')) else 2)
    X=torch.tensor(xx,device='cuda');Y=torch.tensor(yy,device='cuda');W=torch.tensor(masks,device='cuda')
    counts=[kinds.count(i) for i in range(3)];weights=torch.tensor([[.20,.40,.40][k]/counts[k] for k in kinds],device='cuda')
    best=-1;best_qat=-1;history=[];start=time.monotonic();qat=False
    print('Training',model.num_params(),'parameters;',len(items),'examples;',torch.cuda.get_device_name(),flush=True)
    for step in range(1,args.steps+1):
        if step==args.qat_start:
            enable_qat(model);qat=True;print('Enabled int4 weight / output and int8 embedding QAT',flush=True)
        model.train();sel=torch.multinomial(weights,96,replacement=True);opt.zero_grad(set_to_none=True)
        logits,_=model(X[sel]);losses=torch.nn.functional.cross_entropy(logits.reshape(-1,cfg.vocab),Y[sel].reshape(-1),ignore_index=-100,reduction='none').view(96,width)
        loss=(losses*W[sel]).sum()/W[sel].sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if qat:lr=.00025
        else:lr=.002*min(step/100,1)*(.25+.75*(1-min(step/args.qat_start,1)))
        for group in opt.param_groups:group['lr']=lr
        opt.step()
        if step%100==0:print('step',step,'loss',round(loss.item(),5),'seconds',round(time.monotonic()-start,1),flush=True)
        if step%500==0 or step==args.steps:
            report=evaluate(model,dev,tok);report.update(step=step,qat=qat,seconds=time.monotonic()-start)
            (ROOT/'training'/f'dev-{step}.json').write_text(json.dumps(report,indent=2))
            ckpt={'config':asdict(cfg),'state':plain_state(model),'extra':{'step':step,'qat':qat,'dev_accuracy':report['accuracy'],'selection_score':report['selection_score'],'gate':True,'tokenizer_sha256':hashlib.sha256((ROOT/'data/tokenizer.json').read_bytes()).hexdigest()}}
            torch.save(ckpt,ROOT/'training/latest.pt')
            torch.save(ckpt,ROOT/'training'/f'checkpoint-{step}.pt')
            if report['selection_score']>best:best=report['selection_score'];torch.save(ckpt,ROOT/'training/best.pt')
            if qat and report['selection_score']>best_qat:best_qat=report['selection_score'];torch.save(ckpt,ROOT/'training/best-qat.pt')
            history.append({k:v for k,v in report.items() if k!='results'});(ROOT/'training/history.json').write_text(json.dumps(history,indent=2))
            print('DEV',step,'known',report['known_exact'],'/',report['known_count'],'false refusal',report['known_refused'],'unknown',report['unknown_refused'],'/',report['unknown_count'],'QAT',qat,flush=True)
    print('Training complete',round(time.monotonic()-start,1),'seconds',flush=True)
if __name__=='__main__':main()
