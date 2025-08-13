#!/usr/bin/env python3
"""
genome_flow_face_draft1.py

This script is an earlier draft of the genome art generator. It parses a 23andMe raw
genome file, generates a flow‑field texture seeded by the genotype data, and
constructs a face silhouette with simple facial features. Hair is drawn as a
polygon above the head with vertical strands. The nose is a slender trapezoid,
and eyes and lips are stylised but simpler than in the more refined version.

Usage:
    python genome_flow_face_draft1.py /path/to/23andme.txt output_prefix

Outputs (1024x1024 previews):
    <prefix>_flow_var.png                # texture alone
    <prefix>_face_realistic.png          # face with realistic mask, texture, simple features
    <prefix>_face_stylized.png           # face with stylised mask, texture, simple features
    <prefix>_mask_realistic.png          # debug mask on white
    <prefix>_mask_stylized.png           # debug mask on white

Note: A later draft introduces more detailed eyes, a triangular nose, and wavy
hair strands. This file preserves the earlier, less polished design for
reference.
"""

import sys
import os
import hashlib
import random
import numpy as np
from PIL import Image, ImageDraw

# --- Genome parsing and seed/palette utilities ---
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
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    nibbles = []
    for gt in genotypes:
        chars = gt if len(gt) == 2 else gt * 2
        for ch in chars[:2]:
            nibbles.append(mapping.get(ch, 4))
    out = bytearray()
    for i in range(0, len(nibbles), 2):
        a = nibbles[i]
        b = nibbles[i + 1] if i + 1 < len(nibbles) else 0
        out.append((a << 4) | (b & 0xF))
    return bytes(out)

def hash_bytes(b):
    return hashlib.sha256(b).digest()

def palette_from_hash(h, n=6):
    return [(h[i * 3], h[i * 3 + 1], h[i * 3 + 2]) for i in range(n)]

def sex_from_rows(rows):
    has_y = any(ch in ('Y', '24') for ch, _ in rows)
    has_x = any(ch in ('X', '23') for ch, _ in rows)
    if has_y:
        return 'male'
    if has_x:
        return 'female'
    return 'unknown'

# --- Flow‑field texture ---
def flow_field_texture(size, palette, seed, num_lines=1800, steps=120, bg=(0, 0, 0)):
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    w, h = size
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    field = np.random.rand(64, 64, 2) * 2 - 1
    for _ in range(num_lines):
        x = random.random() * w
        y = random.random() * h
        color = random.choice(palette)
        width = random.uniform(0.6, 2.6)
        for _ in range(steps):
            gx = int((x / w) * 63) % 64
            gy = int((y / h) * 63) % 64
            angle = field[gy, gx, 0] * np.pi * 2
            x2 = x + np.cos(angle) * 2.2
            y2 = y + np.sin(angle) * 2.2
            draw.line((x, y, x2, y2), fill=color, width=int(width))
            width = max(0.5, min(3.5, width + (random.random() - 0.5) * 0.22))
            x, y = x2, y2
    return img

# --- Helpers for masks and features ---
SUPERSAMPLE = 4

def build_face_realistic_mask(size, seed, sex_hint='unknown'):
    w, h = size
    W, H = w * SUPERSAMPLE, h * SUPERSAMPLE
    rng = np.random.RandomState((seed ^ 0xA5A5A5A5) & 0xFFFFFFFF)
    if sex_hint == 'male':
        rx = int(W * rng.uniform(0.22, 0.26))
        ry = int(H * rng.uniform(0.30, 0.34))
    elif sex_hint == 'female':
        rx = int(W * rng.uniform(0.26, 0.30))
        ry = int(H * rng.uniform(0.32, 0.36))
    else:
        rx = int(W * rng.uniform(0.24, 0.28))
        ry = int(H * rng.uniform(0.31, 0.35))
    cx, cy = W // 2, int(H * 0.50)
    mask_big = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask_big)
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    return mask_big.resize((w, h), Image.LANCZOS)

def build_face_stylized_mask(size, seed, sex_hint='unknown'):
    w, h = size
    W, H = w * SUPERSAMPLE, h * SUPERSAMPLE
    cx, cy = W // 2, int(H * 0.46)
    rng = np.random.RandomState((seed ^ 0x5EEDFACE) & 0xFFFFFFFF)
    head_rx = int(W * rng.uniform(0.20, 0.26))
    head_ry = int(H * rng.uniform(0.26, 0.32))
    jaw_drop = int(H * rng.uniform(0.10, 0.15))
    chin_half_w = int(head_rx * rng.uniform(0.35, 0.50))
    if sex_hint == 'male':
        jaw_drop = int(jaw_drop * 1.1)
        chin_half_w = int(chin_half_w * 1.1)
    elif sex_hint == 'female':
        head_ry = int(head_ry * 0.95)
    # Flip top arc to avoid horns
    top_arc = []
    for t in np.linspace(np.pi, 0, 180):
        x = cx + head_rx * np.cos(t)
        y = cy - head_ry + (-head_ry) * np.sin(t)
        top_arc.append((x, y))
    jaw_right = (cx + head_rx, cy)
    jaw_left = (cx - head_rx, cy)
    chin_y = cy + jaw_drop
    chin_r = (cx + chin_half_w, chin_y)
    chin_l = (cx - chin_half_w, chin_y)
    outline = top_arc + [jaw_right, chin_r, chin_l, jaw_left]
    mask_big = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask_big)
    d.polygon(outline, fill=255)
    return mask_big.resize((w, h), Image.LANCZOS)

def export_debug_mask(mask_img, outfile, target_size=(1024, 1024), margin=0.08):
    tw, th = target_size
    debug = Image.new("RGB", target_size, (255, 255, 255))
    bbox = mask_img.getbbox()
    if not bbox:
        debug.save(outfile)
        return
    cropped = mask_img.crop(bbox)
    cw, ch = cropped.size
    scale = min((tw * (1 - margin)) / cw, (th * (1 - margin)) / ch)
    ns = (max(1, int(cw * scale)), max(1, int(ch * scale)))
    scaled = cropped.resize(ns, Image.LANCZOS)
    px = (tw - ns[0]) // 2
    py = (th - ns[1]) // 2
    debug.paste((0, 0, 0), box=(px, py, px + ns[0], py + ns[1]), mask=scaled)
    draw = ImageDraw.Draw(debug)
    draw.rectangle([px - 1, py - 1, px + ns[0], py + ns[1]], outline=(30, 30, 30), width=1)
    debug.save(outfile)

def apply_texture_to_mask(texture_img, mask_img, bg_color=(245, 245, 248)):
    out = Image.new("RGB", texture_img.size, bg_color)
    out.paste(texture_img, (0, 0), mask_img)
    return out

def _lighten(color, factor=0.5):
    return tuple(int((1 - factor) * c + factor * 255) for c in color)

def _darken(color, factor=0.5):
    return tuple(int((1 - factor) * c) for c in color)

def draw_hair(img, mask_img, palette, seed, sex_hint='unknown'):
    bbox = mask_img.getbbox()
    if not bbox:
        return
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    cx = (x0 + x1) / 2.0
    rng = random.Random(hash((seed, 'hair')))
    # Hair ratios similar to early draft: male shorter, female longer
    if sex_hint == 'male':
        hair_ratio = rng.uniform(0.12, 0.22)
    elif sex_hint == 'female':
        hair_ratio = rng.uniform(0.28, 0.38)
    else:
        hair_ratio = rng.uniform(0.18, 0.32)
    hair_height = h * hair_ratio
    # Hair colour: darken palette
    hair_color = _darken(palette[3], factor=0.6)
    draw = ImageDraw.Draw(img)
    segments = max(30, int(w * 0.15))
    hair_top_points = []
    head_top_points = []
    for i in range(segments + 1):
        t = i / segments
        x = x0 + t * w
        col_x = int(min(max(int(x), 0), mask_img.width - 1))
        head_top_y = None
        for yy in range(max(0, int(y0) - int(hair_height * 1.2)), min(mask_img.height, int(y1))):
            if mask_img.getpixel((col_x, yy)) > 0:
                head_top_y = yy
                break
        if head_top_y is None:
            continue
        if sex_hint == 'male':
            var = rng.uniform(0.4, 0.9)
        elif sex_hint == 'female':
            var = rng.uniform(0.8, 2.2)
        else:
            var = rng.uniform(0.6, 1.4)
        hair_len = hair_height * var
        hair_top_y = head_top_y - hair_len
        hair_top_points.append((x, hair_top_y))
        head_top_points.append((x, head_top_y))
    if hair_top_points and head_top_points:
        poly = hair_top_points + head_top_points[::-1]
        poly_clipped = [(max(0, min(px, img.width - 1)), max(0, min(py, img.height - 1))) for (px, py) in poly]
        draw.polygon(poly_clipped, fill=hair_color)
        stroke_count = max(10, int(w * 0.1))
        for j in range(stroke_count):
            tt = (j + rng.random()) / stroke_count
            sx = x0 + tt * w
            col_x2 = int(min(max(int(sx), 0), mask_img.width - 1))
            head_y = None
            for yy in range(max(0, int(y0) - int(hair_height * 1.2)), min(mask_img.height, int(y1))):
                if mask_img.getpixel((col_x2, yy)) > 0:
                    head_y = yy
                    break
            if head_y is None:
                continue
            idx = int(tt * (len(hair_top_points) - 1))
            ht_x, ht_y = hair_top_points[idx]
            line_width = max(1, int(w * 0.003))
            draw.line([(sx, ht_y), (sx, head_y)], fill=_darken(hair_color, factor=0.2), width=line_width)

def draw_facial_features(img, mask_img, palette, sex_hint='unknown', seed=None):
    bbox = mask_img.getbbox()
    if not bbox:
        return
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    cx = (x0 + x1) / 2.0
    rng = random.Random(seed) if seed is not None else None
    # Eye parameters
    eye_y = y0 + h * 0.35
    eye_offset_x = w * 0.20
    eye_w = w * 0.12
    eye_h = h * 0.06
    if rng:
        jitter = lambda base, scale: base + scale * (rng.random() - 0.5)
        eye_y = jitter(eye_y, h * 0.02)
        eye_offset_x = jitter(eye_offset_x, w * 0.02)
        eye_w = jitter(eye_w, w * 0.02)
        eye_h = jitter(eye_h, h * 0.02)
    left_eye = [cx - eye_offset_x - eye_w / 2, eye_y - eye_h / 2,
                cx - eye_offset_x + eye_w / 2, eye_y + eye_h / 2]
    right_eye = [cx + eye_offset_x - eye_w / 2, eye_y - eye_h / 2,
                 cx + eye_offset_x + eye_w / 2, eye_y + eye_h / 2]
    draw = ImageDraw.Draw(img)
    sclera_color = _lighten(palette[0], factor=0.7)
    iris_color = _lighten(palette[3], factor=0.3)
    pupil_color = (20, 20, 20)
    draw.ellipse(left_eye, fill=sclera_color)
    draw.ellipse(right_eye, fill=sclera_color)
    iris_w = eye_w * 0.5
    iris_h = eye_h * 0.6
    left_iris = [ (left_eye[0] + left_eye[2]) / 2 - iris_w / 2,
                  (left_eye[1] + left_eye[3]) / 2 - iris_h / 2,
                  (left_eye[0] + left_eye[2]) / 2 + iris_w / 2,
                  (left_eye[1] + left_eye[3]) / 2 + iris_h / 2 ]
    right_iris = [ (right_eye[0] + right_eye[2]) / 2 - iris_w / 2,
                   (right_eye[1] + right_eye[3]) / 2 - iris_h / 2,
                   (right_eye[0] + right_eye[2]) / 2 + iris_w / 2,
                   (right_eye[1] + right_eye[3]) / 2 + iris_h / 2 ]
    draw.ellipse(left_iris, fill=iris_color)
    draw.ellipse(right_iris, fill=iris_color)
    pupil_w = eye_w * 0.25
    pupil_h = eye_h * 0.35
    left_pupil = [ (left_eye[0] + left_eye[2]) / 2 - pupil_w / 2,
                   (left_eye[1] + left_eye[3]) / 2 - pupil_h / 2,
                   (left_eye[0] + left_eye[2]) / 2 + pupil_w / 2,
                   (left_eye[1] + left_eye[3]) / 2 + pupil_h / 2 ]
    right_pupil = [ (right_eye[0] + right_eye[2]) / 2 - pupil_w / 2,
                    (right_eye[1] + right_eye[3]) / 2 - pupil_h / 2,
                    (right_eye[0] + right_eye[2]) / 2 + pupil_w / 2,
                    (right_eye[1] + right_eye[3]) / 2 + pupil_h / 2 ]
    draw.ellipse(left_pupil, fill=pupil_color)
    draw.ellipse(right_pupil, fill=pupil_color)
    highlight_scale = 0.35
    hl_w = pupil_w * highlight_scale
    hl_h = pupil_h * highlight_scale
    left_hl_center_x = (left_pupil[0] + left_pupil[2]) / 2 + pupil_w * 0.15
    left_hl_center_y = (left_pupil[1] + left_pupil[3]) / 2 - pupil_h * 0.15
    left_hl = [left_hl_center_x - hl_w / 2,
               left_hl_center_y - hl_h / 2,
               left_hl_center_x + hl_w / 2,
               left_hl_center_y + hl_h / 2]
    right_hl_center_x = (right_pupil[0] + right_pupil[2]) / 2 + pupil_w * 0.15
    right_hl_center_y = (right_pupil[1] + right_pupil[3]) / 2 - pupil_h * 0.15
    right_hl = [right_hl_center_x - hl_w / 2,
                right_hl_center_y - hl_h / 2,
                right_hl_center_x + hl_w / 2,
                right_hl_center_y + hl_h / 2]
    draw.ellipse(left_hl, fill=(255, 255, 255))
    draw.ellipse(right_hl, fill=(255, 255, 255))
    outline_color = (30, 30, 30)
    outline_width = max(1, int(w * 0.003))
    draw.ellipse(left_eye, outline=outline_color, width=outline_width)
    draw.ellipse(right_eye, outline=outline_color, width=outline_width)
    # Nose: slender trapezoid like early draft
    nose_color = _lighten(palette[2], factor=0.4)
    nose_top_y = y0 + h * 0.45
    nose_bottom_y = y0 + h * 0.62
    nose_top_w = w * 0.05
    nose_bottom_w = w * 0.10
    if rng:
        nose_bottom_y += h * 0.02 * (rng.random() - 0.5)
        nose_top_w += w * 0.01 * (rng.random() - 0.5)
        nose_bottom_w += w * 0.015 * (rng.random() - 0.5)
    nose_poly = [
        (cx - nose_top_w / 2, nose_top_y),
        (cx + nose_top_w / 2, nose_top_y),
        (cx + nose_bottom_w / 2, nose_bottom_y),
        (cx - nose_bottom_w / 2, nose_bottom_y),
    ]
    draw.polygon(nose_poly, fill=nose_color)
    nostril_color = (30, 30, 30)
    nostril_radius_x = w * 0.015
    nostril_radius_y = h * 0.01
    nostril_y = nose_bottom_y - nostril_radius_y * 0.5
    draw.ellipse([
        cx - nose_bottom_w * 0.3 - nostril_radius_x,
        nostril_y - nostril_radius_y,
        cx - nose_bottom_w * 0.3 + nostril_radius_x,
        nostril_y + nostril_radius_y], fill=nostril_color)
    draw.ellipse([
        cx + nose_bottom_w * 0.3 - nostril_radius_x,
        nostril_y - nostril_radius_y,
        cx + nose_bottom_w * 0.3 + nostril_radius_x,
        nostril_y + nostril_radius_y], fill=nostril_color)
    # Lips: two ellipses with dividing line
    lip_color = _lighten(palette[1], factor=0.5)
    lip_y_center = y0 + h * 0.77
    lip_w = w * 0.32
    lip_h_total = h * 0.075
    if rng:
        lip_w += w * 0.02 * (rng.random() - 0.5)
        lip_h_total += h * 0.02 * (rng.random() - 0.5)
    top_lip_h = lip_h_total * 0.45
    top_lip_box = [cx - lip_w / 2, lip_y_center - lip_h_total / 2,
                   cx + lip_w / 2, lip_y_center - lip_h_total / 2 + top_lip_h]
    bottom_lip_h = lip_h_total * 0.55
    bottom_lip_box = [cx - lip_w / 2, lip_y_center - lip_h_total / 2 + top_lip_h,
                      cx + lip_w / 2, lip_y_center + lip_h_total / 2]
    draw.ellipse(top_lip_box, fill=lip_color)
    draw.ellipse(bottom_lip_box, fill=lip_color)
    lip_line_color = (50, 50, 50)
    line_y = lip_y_center - lip_h_total / 2 + top_lip_h
    draw.line([cx - lip_w / 2, line_y, cx + lip_w / 2, line_y], fill=lip_line_color, width=max(1, int(w * 0.015)))

def main():
    if len(sys.argv) < 3:
        print("Usage: python genome_flow_face_draft1.py /path/to/23andme.txt output_prefix")
        sys.exit(1)
    infile, outpref = sys.argv[1], sys.argv[2]
    if not os.path.isfile(infile):
        print("File not found:", infile)
        sys.exit(1)
    rows = parse_23andme_with_chr(infile)
    if not rows:
        print("No SNP rows parsed.")
        sys.exit(1)
    genotypes = [gt for _, gt in rows]
    gen_bytes = genotypes_to_bytes(genotypes)
    h = hash_bytes(gen_bytes)
    seed = int.from_bytes(h[:8], 'big')
    palette = palette_from_hash(h, n=6)
    sex_hint = sex_from_rows(rows)
    size = (1024, 1024)
    # Texture
    texture = flow_field_texture(size, palette, seed, num_lines=1800, steps=120, bg=(0, 0, 0))
    texture.save(f"{outpref}_flow_var.png")
    # Masks
    mask_real = build_face_realistic_mask(size, seed, sex_hint)
    mask_styl = build_face_stylized_mask(size, seed + 1, sex_hint)
    export_debug_mask(mask_real, f"{outpref}_mask_realistic.png", target_size=size)
    export_debug_mask(mask_styl, f"{outpref}_mask_stylized.png", target_size=size)
    # Composite textures
    face_real = apply_texture_to_mask(texture, mask_real, bg_color=(245, 245, 248))
    face_styl = apply_texture_to_mask(texture, mask_styl, bg_color=(245, 245, 248))
    # Draw features and hair
    draw_facial_features(face_real, mask_real, palette, sex_hint, seed=seed)
    draw_facial_features(face_styl, mask_styl, palette, sex_hint, seed=seed + 1)
    draw_hair(face_real, mask_real, palette, seed, sex_hint)
    draw_hair(face_styl, mask_styl, palette, seed + 2, sex_hint)
    face_real.save(f"{outpref}_face_realistic.png")
    face_styl.save(f"{outpref}_face_stylized.png")
    print("Done generating files with prefix", outpref)

if __name__ == "__main__":
    main()