from pathlib import Path
from dataclasses import asdict
import json,hashlib,random,argparse
import torch
from model import SidLM
from quant import export,calibrate_kv,fake_quant_model
from tokenizer import Tokenizer,EOS,ROOT

def export_checkpoint(checkpoint,path,seqs=None):
    checkpoint=Path(checkpoint);path=Path(path);torch.set_num_threads(6);tok=Tokenizer.load();model=SidLM.load(checkpoint).cuda().eval()
    if seqs is None:
        items=json.loads((ROOT/'data/calibration.json').read_text())
        seqs=[tok.prompt(x['question'])+tok.completion(x) for x in items]
    calibration=fake_quant_model(model).cuda().eval()
    kp,vp=calibrate_kv(calibration,seqs,'cuda');info=export(model,path,kp,vp)
    blob=path.read_bytes();fnv=2166136261
    for b in blob:fnv=((fnv^b)*16777619)&0xffffffff
    info.update(parameters=model.num_params(),config=asdict(model.cfg),sha256=hashlib.sha256(blob).hexdigest(),fnv1a=fnv,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),calibration_examples=len(seqs),storage='int4 weights in one byte per weight; int8 input embeddings; int4 KV cache',gate='known=5, unknown=6; no numerical confidence claim')
    info['calibration_sha256']=hashlib.sha256((ROOT/'data/calibration.json').read_bytes()).hexdigest()
    (path.parent/'export-report.json').write_text(json.dumps(info,indent=2))
    return info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,default=ROOT/'training/best-qat.pt');ap.add_argument('--out',type=Path,default=ROOT/'build/MEGAQA.BIN');ap.add_argument('--no-meta',action='store_true');args=ap.parse_args()
    info=export_checkpoint(args.checkpoint,args.out)
    blob=args.out.read_bytes();fnv=info['fnv1a']
    if not args.no_meta:(ROOT/'native/model_meta.h').write_text(f'#define QA_MODEL_BYTES {len(blob)}UL\n#define QA_MODEL_FNV 0x{fnv:08x}UL\n')
    print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__':main()
