"""Minimal offline FAT32 access for a PRIVATE Xemu test image, never a device.
No deletion or replacement. Adds files to the root of a copied Xemu SD image.
"""
from pathlib import Path
import struct,sys
class FAT:
    def __init__(self,path,write=False):
        self.path=Path(path).resolve();assert self.path.is_file() and self.path.suffix=='.img'
        self.f=self.path.open('r+b' if write else 'rb');m=self.f.read(512)
        self.base=struct.unpack_from('<I',m,454)[0]*512;self.f.seek(self.base);b=self.f.read(512)
        assert m[510:512]==b'\x55\xaa' and m[450] in (11,12)
        assert struct.unpack_from('<H',b,11)[0]==512
        self.spc=b[13];self.cs=self.spc*512;self.res=struct.unpack_from('<H',b,14)[0];self.nf=b[16];self.fatsz=struct.unpack_from('<I',b,36)[0];self.root=struct.unpack_from('<I',b,44)[0]
        self.fat_at=self.base+self.res*512;self.data_at=self.fat_at+self.nf*self.fatsz*512
        self.cluster_limit=2+(struct.unpack_from('<I',b,32)[0]-self.res-self.nf*self.fatsz)//self.spc
        assert self.nf==2 and self.spc and self.cluster_limit*4<=self.fatsz*512
        self.f.seek(self.fat_at);self.fat=bytearray(self.f.read(self.fatsz*512))
    def get(self,c):return struct.unpack_from('<I',self.fat,c*4)[0]&0xfffffff
    def at(self,c):return self.data_at+(c-2)*self.cs
    def chain(self,c):
        seen=set()
        while 2<=c<0xffffff8:
            assert c not in seen and c<self.cluster_limit;seen.add(c);yield c;c=self.get(c)
    def entries(self):
        for c in self.chain(self.root):
            self.f.seek(self.at(c));data=self.f.read(self.cs)
            for i in range(0,self.cs,32):
                e=data[i:i+32]
                if not e[0]:return
                if e[0]==229 or e[11] in (15,8):continue
                name=e[:8].decode('ascii').rstrip();ext=e[8:11].decode('ascii').rstrip()
                yield name+('.'+ext if ext else ''),self.at(c)+i,e
    def read(self,name):
        _,_,e=next(v for v in self.entries() if v[0]==name.upper())
        c=struct.unpack_from('<H',e,26)[0]|struct.unpack_from('<H',e,20)[0]<<16;size=struct.unpack_from('<I',e,28)[0]
        out=bytearray()
        for c in self.chain(c):self.f.seek(self.at(c));out.extend(self.f.read(self.cs))
        return bytes(out[:size])
    def add(self,name,data):
        assert name.upper() not in [v[0] for v in self.entries()],name
        a,b=name.upper().split('.');assert len(a)<=8 and len(b)<=3
        n=(len(data)+self.cs-1)//self.cs
        free=[c for c in range(2,self.cluster_limit) if self.get(c)==0][:n];assert len(free)==n
        slot=None
        for c in self.chain(self.root):
            self.f.seek(self.at(c));d=self.f.read(self.cs)
            for i in range(0,self.cs,32):
                if d[i] in (0,229):slot=self.at(c)+i;break
            if slot is not None:break
        assert slot is not None,'Root directory full'
        for i,c in enumerate(free):
            struct.pack_into('<I',self.fat,c*4,free[i+1] if i+1<n else 0xfffffff)
            self.f.seek(self.at(c));self.f.write(data[i*self.cs:(i+1)*self.cs].ljust(self.cs,b'\0'))
        for i in range(self.nf):self.f.seek(self.fat_at+i*self.fatsz*512);self.f.write(self.fat)
        e=bytearray(32);e[:11]=a.ljust(8).encode()+b.ljust(3).encode();e[11]=32;struct.pack_into('<H',e,20,free[0]>>16);struct.pack_into('<H',e,26,free[0]&65535);struct.pack_into('<I',e,28,len(data));self.f.seek(slot);self.f.write(e);self.f.flush()
        assert self.read(name)==data
    def mkdir(self,name):
        assert 1<=len(name)<=8 and name.isalnum()
        assert name.upper() not in [v[0] for v in self.entries()]
        temporary=name+'.DIR';self.add(temporary,bytes(self.cs))
        _,slot,raw=next(e for e in self.entries() if e[0]==temporary.upper());entry=bytearray(raw)
        c=struct.unpack_from('<H',entry,26)[0]|struct.unpack_from('<H',entry,20)[0]<<16
        entry[:11]=name.upper().ljust(11).encode();entry[11]=16;entry[28:32]=bytes(4)
        d=bytearray(self.cs)
        for at,n,cluster in [(0,'.',c),(32,'..',self.root)]:
            d[at:at+11]=n.ljust(11).encode();d[at+11]=16
            struct.pack_into('<H',d,at+20,cluster>>16);struct.pack_into('<H',d,at+26,cluster&65535)
        self.f.seek(self.at(c));self.f.write(d);self.f.seek(slot);self.f.write(entry);self.f.flush();return c
    def close(self):self.f.close()
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    fat=FAT(root/'test-sd.img',True)
    model=(root/'build/MEGAQA.BIN').read_bytes()
    if 'MEGAQA.BIN' not in [v[0] for v in fat.entries()]:fat.add('MEGAQA.BIN',model)
    assert fat.read('MEGAQA.BIN')==model
    glow=(root/'build/GLOW.BIN').read_bytes()
    if 'GLOW.BIN' not in [v[0] for v in fat.entries()]:fat.add('GLOW.BIN',glow)
    assert fat.read('GLOW.BIN')==glow
    if 'TINYBR61' not in [v[0] for v in fat.entries()]:
        fat.root=fat.mkdir('TINYBR61');fat.add('MEGAQA.BIN',model);fat.add('GLOW.BIN',glow)
        fat.add('TINYBR61.D81',(root/'build/TINYBR61.D81').read_bytes())
    print('Private SD files:',[e[0] for e in fat.entries()]);fat.close()
