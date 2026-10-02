"""Build TINYBR61.D81; MEGAQA.BIN is a separate file in the same SD folder."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def offset(track, sector):
    assert 1 <= track <= 80 and 0 <= sector < 40
    return ((track - 1) * 40 + sector) * 256


def make_disk(files):
    image = bytearray(819200)
    free = {t: [True] * 40 for t in range(1, 81)}
    for s in range(4):
        free[40][s] = False
    header = offset(40, 0)
    image[header:header + 3] = bytes([40, 3, 0x44])
    image[header + 4:header + 29] = b'TINYBR61'.ljust(18, b'\xa0') + b'QA\xa03D\xa0\xa0'
    sectors = [(t, s) for t in sorted(free, key=lambda t: abs(t - 40))
               if t != 40 for s in range(40)]
    assert 1 <= len(files) <= 8
    used = 0
    for i, (name, data) in enumerate(files):
        chunks = [data[p:p + 254] for p in range(0, len(data), 254)]
        assert chunks and len(name) <= 16
        chain = sectors[used:used + len(chunks)]
        assert len(chain) == len(chunks), 'Disk full'
        used += len(chain)
        for j, ((t, s), chunk) in enumerate(zip(chain, chunks)):
            free[t][s] = False
            at = offset(t, s)
            link = chain[j + 1] if j + 1 < len(chain) else (0, len(chunk) + 1)
            image[at:at + 2] = bytes(link)
            image[at + 2:at + 2 + len(chunk)] = chunk
        at = offset(40, 3) + i * 32
        image[at + 2:at + 5] = bytes([0x82, *chain[0]])
        image[at + 5:at + 21] = name.encode('ascii').ljust(16, b'\xa0')
        image[at + 30:at + 32] = len(chain).to_bytes(2, 'little')
        print(f'{name:16} {len(data):6} bytes, {len(chain)} blocks')
    image[offset(40, 3):offset(40, 3) + 2] = b'\x00\xff'
    for sector, start in [(1, 1), (2, 41)]:
        at = offset(40, sector)
        image[at:at + 7] = bytes([40, 2, 0x44, 0xbb, 81, 65, 0xc0]) if sector == 1 else bytes([0, 255, 0x44, 0xbb, 81, 65, 0xc0])
        for j in range(40):
            flags = free[start + j]
            entry = at + 16 + j * 6
            image[entry] = sum(flags)
            image[entry + 1:entry + 6] = sum(1 << s for s, yes in enumerate(flags) if yes).to_bytes(5, 'little')
    # Independently follow every directory/file chain and compare the source PRG.
    occupied = {(40, s) for s in range(4)}
    for i, (name, source) in enumerate(files):
        at = offset(40, 3) + i * 32
        t, s = image[at + 3:at + 5]
        restored = bytearray()
        blocks = 0
        while t:
            assert (t, s) not in occupied, 'Cross-linked disk sectors'
            occupied.add((t, s))
            assert not free[t][s]
            at = offset(t, s)
            t, s = image[at:at + 2]
            restored.extend(image[at + 2:at + 256 if t else at + 1 + s])
            blocks += 1
        assert restored == source, name
        entry = offset(40, 3) + i * 32
        assert image[entry + 5:entry + 21].rstrip(b'\xa0') == name.encode('ascii')
        assert blocks == int.from_bytes(image[entry + 30:entry + 32], 'little')
    assert len(occupied) + sum(sum(flags) for flags in free.values()) == 3200
    return image

if __name__=='__main__':
    (HERE/'build/TINYBR61.D81').write_bytes(make_disk([
        ('TINYBR61',(HERE/'native/build/tinybr61.prg').read_bytes())]))



