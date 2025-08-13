#!/usr/bin/env python3

import sys, os, hashlib, random
import numpy as np
from PIL import Image, ImageDraw

# -------------------------------
# Parse + seed + palette + sex
# -------------------------------
def parse_23andme_with_chr(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                chrom = parts[1].upper()
                gt = parts[-1].upper()
                if all(ch in 'ACGT-' for ch in gt):
                    rows.append((chrom, gt))
    return rows

def genotypes_to_bytes(genotypes):
    mapping = {'A':0,'C':1,'G':2,'T':3,'-':4}
    nibbles = []
    for gt in genotypes:
        chars = gt if len(gt)==2 else gt*2
        for ch in chars[:2]:
            nibbles.append(mapping.get(ch,4))
    out = bytearray()
    for i in range(0,len(nibbles),2):
        a = nibbles[i]
        b = nibbles[i+1] if i+1 < len(nibbles) else 0
        out.append((a<<4)|(b&0xF))
    return bytes(out)

def hash_bytes(b): return hashlib.sha256(b).digest()
def palette_from_hash(h,n=6): return [(h[i*3],h[i*3+1],h[i*3+2]) for i in range(n)]

def sex_from_rows(rows):
    has_y = any(ch in ('Y','24') for ch,_ in rows)
    has_x = any(ch in ('X','23') for ch,_ in rows)
    if has_y: return 'male'
    if has_x: return 'female'
    return 'unknown'

# -------------------------------
# Flow-field texture (variable thickness)
# -------------------------------
def flow_field_texture(size, palette, seed, num_lines=1800, steps=120, bg=(0,0,0)):
    random.seed(seed); np.random.seed(seed & 0xFFFFFFFF)
    w,h = size
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    field = np.random.rand(64,64,2)*2 - 1
    for _ in range(num_lines):
        x,y = random.random()*w, random.random()*h
        color = random.choice(palette)
        width = random.uniform(0.6, 2.6)
        for _ in range(steps):
            gx = int((x/w)*63) % 64
            gy = int((y/h)*63) % 64
            angle = field[gy, gx, 0] * np.pi * 2
            x2 = x + np.cos(angle)*2.2
            y2 = y + np.sin(angle)*2.2
            draw.line((x,y,x2,y2), fill=color, width=int(width))
            width = max(0.5, min(3.5, width + (random.random()-0.5)*0.22))
            x,y = x2,y2
    return img

def apply_texture_to_mask(texture_img, mask_img, bg_color=(6,6,8)):
    out = Image.new("RGB", texture_img.size, bg_color)
    out.paste(texture_img, (0,0), mask_img)
    return out

# -------------------------------
# Main
# -------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python genome_flow_face_fixed.py /path/to/23andme.txt output_prefix")
        sys.exit(1)
    infile, outpref = sys.argv[1], sys.argv[2]
    if not os.path.isfile(infile):
        print("File not found:", infile); sys.exit(1)

    rows = parse_23andme_with_chr(infile)
    if not rows:
        print("No SNP rows parsed."); sys.exit(1)

    genotypes = [gt for _,gt in rows]
    gen_bytes = genotypes_to_bytes(genotypes)
    h = hash_bytes(gen_bytes)
    seed = int.from_bytes(h[:8], 'big')
    palette = palette_from_hash(h, n=6)
    sex_hint = sex_from_rows(rows)
    print("Inferred sex:", sex_hint)

    size = (1024, 1024)

    # 1) Flow texture (reference)
    texture = flow_field_texture(size, palette, seed, num_lines=1800, steps=120, bg=(0,0,0))
    texture.save(f"{outpref}_flow_var.png", dpi=(300, 300))
    print("Done.")

if __name__ == "__main__":
    main()
