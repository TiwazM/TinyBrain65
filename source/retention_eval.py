"""Native integer checkpoint scoring; deliberately never loads the fresh test."""
from pathlib import Path
import argparse,hashlib,json
from evaluate_pc import Core
from tokenizer import ROOT
from train_pairs import score_report

def evaluate_rows(core,rows):
    results=[]
    for item in rows:
        r=core.answer(item['question']);r.update(item)
        r['exact']=r['abstained'] if item['id']=='unknown' else not r['abstained'] and r['actual']==item['answer'];results.append(r)
    known=[r for r in results if r['id']!='unknown'];unknown=[r for r in results if r['id']=='unknown']
    return {'count':len(results),'exact':sum(r['exact'] for r in results),'known_count':len(known),'known_exact':sum(r['exact'] for r in known),'known_refused':sum(r['abstained'] for r in known),'unknown_count':len(unknown),'unknown_refused':sum(r['abstained'] for r in unknown),'results':results}

def evaluate_candidate(path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);core=Core(path)
    dev=score_report(evaluate_rows(core,json.loads((ROOT/'data/dev.json').read_text())))
    retention=evaluate_rows(core,json.loads((ROOT/'data/retention-checks.json').read_text()))
    essential=[r for r in retention['results'] if r['essential']]
    losses=[r for r in retention['results'] if any(h['change']=='lost-in-v5' for h in r['history'])]
    gains=[r for r in retention['results'] if any(h['change']=='gained-in-v5' for h in r['history'])]
    retention.update(essential_count=len(essential),essential_exact=sum(r['exact'] for r in essential),formerly_lost_count=len(losses),formerly_lost_restored=sum(r['exact'] for r in losses),v5_gains_count=len(gains),v5_gains_kept=sum(r['exact'] for r in gains))
    model_hash=hashlib.sha256(core.blob).hexdigest()
    for name,report in [('integer-development',dev),('integer-retention',retention)]:
        report['model_sha256']=model_hash;(out/(name+'.json')).write_text(json.dumps(report,indent=2))
    rmacro=.65*retention['known_exact']/retention['known_count']+.35*retention['unknown_refused']/retention['unknown_count']
    summary={'model_sha256':model_hash,'development_score':dev['selection_score'],'development_known_exact':dev['known_exact'],'development_unknown_refused':dev['unknown_refused'],'retention_exact':retention['exact'],'retention_count':retention['count'],'essential_exact':retention['essential_exact'],'essential_count':len(essential),'formerly_lost_restored':retention['formerly_lost_restored'],'v5_gains_kept':retention['v5_gains_kept'],'selection_score':.7*dev['selection_score']+.3*rmacro,'evaluator':'Actual deployed C integer core with int4 KV cache; no fresh test used.'}
    (out/'integer-summary.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();print(json.dumps(evaluate_candidate(args.model,args.out),indent=2))
