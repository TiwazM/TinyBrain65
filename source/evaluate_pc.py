"""Evaluate the actual exported integer model, including its small KV cache."""
import argparse,ctypes as C,hashlib,json,random,time
from pathlib import Path
import numpy as np
from tokenizer import Tokenizer,EOS,KNOWN,UNKNOWN,FIRST_TEXT,REFUSAL,ROOT

class Core:
    def __init__(self,blob_path=None):
        self.lib=C.CDLL(str(ROOT/'native/build/megaqa.dll'));self.blob=Path(blob_path or ROOT/'build/MEGAQA.BIN').read_bytes();self.buffer=C.create_string_buffer(self.blob)
        self.lib.sidlm_load.argtypes=[C.c_void_p,C.c_uint32];self.lib.sidlm_load.restype=C.c_int32
        self.lib.sidlm_advance.argtypes=[C.c_uint16];self.lib.sidlm_logits.argtypes=[C.POINTER(C.c_float),C.c_void_p]
        self.lib.qa_tokenize.argtypes=[C.c_char_p,C.POINTER(C.c_uint16),C.c_uint16];self.lib.qa_tokenize.restype=C.c_int16
        self.lib.qa_piece.argtypes=[C.c_uint16];self.lib.qa_piece.restype=C.c_char_p
        assert self.lib.sidlm_load(self.buffer,len(self.blob))==len(self.blob)
        self.out=np.zeros(1024,np.float32);self.ptr=self.out.ctypes.data_as(C.POINTER(C.c_float));self.tok=Tokenizer.load()
    def tokenize(self,q):
        ids=(C.c_uint16*80)();n=self.lib.qa_tokenize(q.encode('ascii'),ids,80)
        return list(ids[:n]) if n>=0 else None
    def answer(self,q):
        prompt=self.tokenize(q)
        if prompt is None:raise ValueError('Question too long')
        assert prompt==self.tok.prompt(q),(q,prompt,self.tok.prompt(q));self.lib.sidlm_reset()
        for i in prompt:assert self.lib.sidlm_advance(i)==0
        self.lib.sidlm_logits(self.ptr,None);assert np.isfinite(self.out).all()
        gate_margin=float(self.out[KNOWN]-self.out[UNKNOWN]);abstained=gate_margin<0
        if abstained:return {'actual':REFUSAL,'ids':[],'prompt_ids':prompt,'ended':True,'abstained':True,'gate_margin':gate_margin,'gate_token':UNKNOWN}
        assert self.lib.sidlm_advance(KNOWN)==0
        ids=[];ended=False
        for _ in range(40):
            self.lib.sidlm_logits(self.ptr,None);assert np.isfinite(self.out).all();self.out[KNOWN:FIRST_TEXT]=-np.inf;i=int(self.out[4:].argmax())+4
            if i==EOS:ended=True;break
            ids.append(i);assert self.lib.sidlm_advance(i)==0
        return {'actual':self.tok.decode(ids),'ids':ids,'prompt_ids':prompt,'ended':ended,'abstained':False,'gate_margin':gate_margin,'gate_token':KNOWN}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='dev',choices=['dev','test','train','challenge','user']);args=ap.parse_args()
    core=Core()
    if args.split in ['challenge','user']:
        facts={r['id']:r for r in json.loads((ROOT/'data/knowledge.json').read_text())}
        file='challenge_questions.json' if args.split=='challenge' else 'user-regressions.json'
        items=[{'id':i,'question':q,'answer':facts[i]['answer']} for i,q in json.loads((ROOT/file).read_text())]
        taught={r['question'] for r in json.loads((ROOT/'data/train.json').read_text())}
        # Historical challenge cases are intentionally rehearsed by V6 retention.
    else:items=json.loads((ROOT/'data'/f'{args.split}.json').read_text())
    start=time.monotonic();results=[]
    if args.split=='train':items=random.Random(65).sample(items,100)
    for j,item in enumerate(items):
        answer=core.answer(item['question']);answer.update(item);answer['exact']=answer['abstained'] if item['id']=='unknown' else answer['actual']==item['answer'];results.append(answer)
        if not answer['exact']:print('MISS',item['question'],'=>',answer['actual'],'expected',item['answer'],flush=True)
        if j%50==0:print(j+1,'/',len(items),flush=True)
    notes={'dev':'Development set used for model selection.', 'test':'Historical regression; successful cases are deliberately rehearsed in V6.', 'challenge':'Historical regression; successful cases are deliberately rehearsed in V6. Only benchmark-retention-frozen is fresh for V6.', 'user':'User regression cases deliberately rehearsed in V6.', 'train':'Fixed sample of 100 training examples.'}
    notes['user']='User screenshots and contrast cases. These informed training and are regression checks, not independent evaluation.'
    report={'split':args.split,'count':len(items),'exact':sum(x['exact'] for x in results),'seconds':time.monotonic()-start,'model_sha256':hashlib.sha256(core.blob).hexdigest(),'note':notes[args.split]+' Exact string matches of taught facts, not broad programming accuracy.','results':results}
    known=[r for r in results if r['id']!='unknown'];unknown=[r for r in results if r['id']=='unknown']
    report.update(known_exact=sum(r['exact'] for r in known),known_count=len(known),known_refused=sum(r['abstained'] for r in known),unknown_refused=sum(r['abstained'] for r in unknown),unknown_count=len(unknown))
    (ROOT/'build'/f'pc-{args.split}.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='results'},flush=True)
if __name__=='__main__':main()
