"""Lock an experimental candidate, then open the frozen tests once.

No trial passed the conservative replacement gate. A trial build may still
be useful, but it must not be described as a regression-free replacement.
"""
from datetime import datetime,timezone
import hashlib,json,shutil
from tokenizer import ROOT
from evaluate_pc import Core
from retention_eval import evaluate_rows

def read(p):return json.loads((ROOT/p).read_text())
def save(p,v):(ROOT/p).write_text(json.dumps(v,indent=2))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
    assert not (ROOT/'build/hardware-selection.json').exists()
    candidates=[]
    for trial in ['new32','new16','expanded-preserve']:
        for r in read('trials/'+trial+'/history.json'):candidates.append(dict(r,trial=trial))
    eligible=[r for r in candidates if r['eligible']]
    exploratory=[r for r in candidates if r['retained']==699 and r['user_exact']==6 and r['old_known']>=893 and r['new_refused']>=10 and r['old_refused']>=384]
    pool=eligible or exploratory;assert pool,'No usable experimental candidate'
    selected=max(pool,key=lambda r:(r['new_score'],-r['old_losses'],r['old_score']))
    path='trials/'+selected['trial']+f"/step-{selected['step']:04d}"
    report=dict(selected=selected,strict_replacement_gate_passed=bool(eligible),status='experimental candidate; keep published V6',selection_time_utc=datetime.now(timezone.utc).isoformat(),fresh_test_not_evaluated_at_selection=True,fresh_sha256=sha('data/hardware-test.json'),all_candidates=candidates,exploratory_rule='If no strict candidate: require 699/699 retention, 6/6 screenshots, old known >=893, new-dev refusals >=10/12, old refusals >=384/398; maximize hardware dev score, then minimize old losses. Rule set after development experiments, before any fresh test evaluation.')
    save('build/hardware-selection.json',report)
    for name in ['MEGAQA.BIN','export-report.json']:
        shutil.copyfile(ROOT/path/name,ROOT/'build'/name)
    shutil.copyfile(ROOT/path/'checkpoint.pt',ROOT/'training/best-qat.pt')
    meta=read('build/export-report.json');assert meta['sha256']==selected['model_sha256'] and meta['parameters']==375600 and meta['bytes']==515638
    (ROOT/'native/model_meta.h').write_text(f"#define QA_MODEL_BYTES {meta['bytes']}UL\n#define QA_MODEL_FNV 0x{meta['fnv1a']:08x}UL\n")
    fresh=read('data/hardware-test.json');historical=read('benchmark-retention-frozen.json')['items']
    historical=[dict(r,id='unknown' if r['unknown'] else r['id']) for r in historical]
    actual_train={r['question'] for f in ['data/train.json','data/hardware-train.json'] for r in read(f)}
    assert not any(r['question'] in actual_train for r in fresh)
    old_overlap=[r['question'] for r in historical if r['question'] in actual_train]
    comparisons={}
    for label,blob in [('v6','build/V6.BIN'),('v61','build/MEGAQA.BIN')]:
        core=Core(ROOT/blob);reports={}
        for name,items in [('hardware-fresh',fresh),('old-historical',historical),('screenshots',read('data/hardware-user-regressions.json'))]:
            reports[name]=evaluate_rows(core,items)
            save('build/'+label+'-'+name+'.json',reports[name])
        comparisons[label]={name:{k:v for k,v in r.items() if k!='results'} for name,r in reports.items()}
        comparisons[label]['sha256']=sha(blob)
    changes={}
    for name in ['hardware-fresh','old-historical']:
        before=read('build/v6-'+name+'.json')['results'];after=read('build/v61-'+name+'.json')['results']
        changes[name]={'gained':[dict(question=a['question'],before=a['actual'],after=b['actual']) for a,b in zip(before,after) if not a['exact'] and b['exact']], 'lost':[dict(question=a['question'],before=a['actual'],after=b['actual']) for a,b in zip(before,after) if a['exact'] and not b['exact']]}
    output=dict(comparisons=comparisons,changes=changes,old_historical_training_overlap=old_overlap,note='Exact taught-answer comparisons, not general factual accuracy. New test was frozen before training and evaluated only after selection. Older suites have historical exposure; protected and screenshot cases are rehearsed. Selection gate failed; this is a trial, not a guaranteed upgrade.')
    save('build/hardware-comparison.json',output)
    print('SELECTED',selected,flush=True);print(json.dumps(comparisons,indent=2));print('OLD LOSSES',changes['old-historical']['lost'])
if __name__=='__main__':main()
