"""Small ASCII word-piece tokenizer, mirrored by the native C implementation."""
from collections import Counter
from pathlib import Path
import json,re,unicodedata
PAD,BOS,QUESTION,ANSWER,EOS,KNOWN,UNKNOWN=range(7)
FIRST_TEXT=7
REFUSAL='i do not know this one.'
ROOT=Path(__file__).resolve().parent
def normalize(text):
    text=text.replace('\u2019',"'").replace('\u2018',"'").replace('\u201c','"').replace('\u201d','"').replace('\u2013','-').replace('\u2014','-')
    return ' '.join(unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode().lower().split())
class Tokenizer:
    def __init__(self,tokens):
        self.tokens=tokens;self.by_first={}
        for i,t in enumerate(tokens[FIRST_TEXT:],FIRST_TEXT):self.by_first.setdefault(t[0],[]).append((t,i))
        for items in self.by_first.values():items.sort(key=lambda p:(-len(p[0]),p[1]))
    @classmethod
    def train(cls,texts,size=1024):
        count=Counter()
        for text in texts:
            for piece in re.findall(r' ?[a-z0-9]+| ?[^a-z0-9\s]',normalize(text)):
                if len(piece)>1:count[piece]+=1
        tokens=['<pad>','<bos>','<question>','<answer>','<end>','<known>','<unknown>']+[chr(i) for i in range(32,127)]
        tokens += [t for t,n in sorted(count.items(),key=lambda p:(-p[1]*len(p[0]),p[0])) if t not in tokens][:size-len(tokens)]
        return cls(tokens)
    @classmethod
    def load(cls):return cls(json.loads((ROOT/'data/tokenizer.json').read_text()))
    def save(self):
        (ROOT/'data/tokenizer.json').write_text(json.dumps(self.tokens,indent=2))
        # The native binary uses the same strings and greedy matching order.
        blob=b'';offsets=[]
        for t in self.tokens:offsets.append(len(blob));blob+=t.encode('ascii')+b'\0'
        code='#ifndef QA_TOKENS_H\n#define QA_TOKENS_H\n#include <stdint.h>\n'
        code+=f'#define QA_VOCAB {len(self.tokens)}\n'
        code+='static const uint16_t token_offsets[]={'+','.join(map(str,offsets))+'};\n'
        code+='static const unsigned char token_text[]={'+','.join(map(str,blob))+'};\n#endif\n'
        (ROOT/'native/tokens.h').write_text(code)
    def encode(self,text):
        text=normalize(text);ids=[];pos=0
        while pos<len(text):
            for piece,i in self.by_first[text[pos]]:
                if text.startswith(piece,pos):ids.append(i);pos+=len(piece);break
        return ids
    def decode(self,ids):return ''.join(self.tokens[i] for i in ids if i>=FIRST_TEXT)
    def prompt(self,q):return [BOS,QUESTION]+self.encode(q)+[ANSWER]
    def completion(self,item):return [UNKNOWN,EOS] if item['id']=='unknown' else [KNOWN]+self.encode(item['answer'])+[EOS]
