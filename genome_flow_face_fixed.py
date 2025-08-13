#!/usr/bin/env python3
"""
genome_flow_face_fixed.py

Usage:
    python genome_flow_face_fixed.py /path/to/23andme.txt output_prefix

Outputs (1024x1024 previews):
    <prefix>_flow_var.png                # flow texture alone
    <prefix>_mask_realistic.png          # centered debug mask on white (realistic)
    <prefix>_mask_stylized.png           # centered debug mask on white (stylized)
    <prefix>_face_realistic.png          # texture clipped to realistic mask
    <prefix>_face_stylized.png           # texture clipped to stylized mask

This script generates generative art from a genome input. It
computes a color palette and seed from the genome file, draws a
flow-field texture, builds two different face silhouette masks
(realistic and stylized), and then composites the texture inside the
masks. Debug images of just the silhouette masks are exported on a
white background to verify that the masks look like faces.

"""

import sys
import os
import hashlib
import random
import math
import numpy as np
from PIL import Image, ImageDraw

# ------------------------------------------------------------------
# Genome parsing, seed and palette derivation, and sex inference
# ------------------------------------------------------------------
def parse_23andme_with_chr(path):
    """Parse a 23andMe-style raw text file.

    Each line is assumed to contain at least four whitespace-separated
    fields: rsid, chromosome, position, genotype. Header lines
    starting with '#' are ignored. Only lines where the genotype
    contains only the letters A, C, G, T, or '-' are kept.

    Returns a list of (chromosome, genotype) tuples.
    """
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
    """Pack genotype strings into a bytes object.

    Each genotype is two characters (or one character repeated) from
    A, C, G, T, or '-'. These are mapped to nibble values 0-4 using
    a simple mapping. Two nibbles are packed into each output byte.
    """
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
    """Compute a SHA-256 digest of the input bytes."""
    return hashlib.sha256(b).digest()

def palette_from_hash(h, n=6):
    """Create an RGB palette of length n from a hash digest."""
    return [(h[i * 3], h[i * 3 + 1], h[i * 3 + 2]) for i in range(n)]

def sex_from_rows(rows):
    """Infer sex from genome rows.

    If any row has chromosome Y (or '24' for some files), returns
    'male'. If there are rows on chromosome X (or '23'), returns
    'female'. Otherwise returns 'unknown'.
    """
    has_y = any(ch in ('Y', '24') for ch, _ in rows)
    has_x = any(ch in ('X', '23') for ch, _ in rows)
    if has_y:
        return 'male'
    if has_x:
        return 'female'
    return 'unknown'

# ------------------------------------------------------------------
# Flow-field texture generation
# ------------------------------------------------------------------
def flow_field_texture(size, palette, seed, num_lines=1800, steps=120, bg=(0, 0, 0)):
    """Generate a flow-field texture as an RGB image.

    A vector field is generated randomly (deterministically using the seed).
    A number of lines are then traced through the field. Each line is
    drawn using a color from the palette and has variable thickness
    modulated by random noise. Returns a PIL Image.
    """
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    w, h = size
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    # Random vector field on a coarse grid
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
            # Slightly vary the width
            width = max(0.5, min(3.5, width + (random.random() - 0.5) * 0.22))
            x, y = x2, y2
    return img

# ------------------------------------------------------------------
# Geometry helpers for parametric curve sampling
# ------------------------------------------------------------------
def sample_cubic_bezier(p0, p1, p2, p3, n=80):
    """Sample n points along a cubic Bézier curve defined by control points."""
    t = np.linspace(0, 1, n)
    one_t = 1 - t
    x = (one_t ** 3) * p0[0] + 3 * (one_t ** 2) * t * p1[0] + 3 * one_t * (t ** 2) * p2[0] + (t ** 3) * p3[0]
    y = (one_t ** 3) * p0[1] + 3 * (one_t ** 2) * t * p1[1] + 3 * one_t * (t ** 2) * p2[1] + (t ** 3) * p3[1]
    return list(zip(x, y))

def sample_ellipse_arc(cx, cy, rx, ry, a0, a1, steps=160):
    """Sample points along an elliptical arc from angle a0 to a1."""
    ang = np.linspace(a0, a1, steps)
    x = cx + rx * np.cos(ang)
    y = cy + ry * np.sin(ang)
    return list(zip(x, y))

def mirror_x(points, about_x):
    """Mirror a list of points across a vertical line x = about_x."""
    return [(about_x - (px - about_x), py) for (px, py) in points]

# ------------------------------------------------------------------
# Face mask construction with supersampling for smooth edges
# ------------------------------------------------------------------
SUPERSAMPLE = 4  # factor by which to draw masks for anti-aliasing

def build_face_realistic_mask(size, seed, sex_hint='unknown'):
    """Build an anti-aliased 'realistic' head silhouette mask.

    The mask is drawn at a higher resolution (size * SUPERSAMPLE) and
    then downsampled for smooth edges. The top of the head is an
    elliptical arc, and the jawlines are cubic Bézier curves. Control
    points are chosen with random variations influenced by the seed
    and the inferred sex.

    Returns a single-channel (L-mode) PIL Image.
    """
    # Use an ellipse to approximate the head silhouette for a smoother, more hand‑drawn look.
    w, h = size
    W, H = w * SUPERSAMPLE, h * SUPERSAMPLE
    rng = np.random.RandomState((seed ^ 0xA5A5A5A5) & 0xFFFFFFFF)
    # Determine radii based on sex: females tend to have wider heads and slightly rounder proportions
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
    mask = mask_big.resize((w, h), Image.LANCZOS)
    return mask

def build_face_stylized_mask(size, seed, sex_hint='unknown'):
    """Build an anti-aliased stylized head mask.

    The stylized head uses a simple ellipse for the skull and a trapezoid
    shape for the jaw/chin. Random variations are applied to widths and
    drops. The mask is drawn at a higher resolution and then
    downsampled for smooth edges.

    Returns a single-channel (L-mode) PIL Image.
    """
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
    # Flip the top arc vertically by using a negative ry so the dome points upward
    top_arc = sample_ellipse_arc(cx, cy - head_ry, head_rx, -head_ry, np.pi, 0.0, steps=180)
    jaw_right = (cx + head_rx, cy)
    jaw_left = (cx - head_rx, cy)
    chin_y = cy + jaw_drop
    chin_r = (cx + chin_half_w, chin_y)
    chin_l = (cx - chin_half_w, chin_y)
    outline = top_arc + [jaw_right, chin_r, chin_l, jaw_left]
    mask_big = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask_big)
    d.polygon(outline, fill=255)
    mask = mask_big.resize((w, h), Image.LANCZOS)
    return mask

# ------------------------------------------------------------------
# Debug mask exporter: center and scale mask on white background
# ------------------------------------------------------------------
def export_debug_mask(mask_img, outfile, target_size=(1024, 1024), margin=0.08):
    """Export a mask for visual debugging.

    The mask image (single channel) is cropped to its content,
    scaled up to fill most of the target size (with margin), centered
    on a white background, and saved as an RGB image. A thin border
    is drawn around the silhouette for clarity.
    """
    tw, th = target_size
    debug = Image.new("RGB", target_size, (255, 255, 255))
    bbox = mask_img.getbbox()
    if not bbox:
        debug.save(outfile)
        return
    # Crop to nonzero region
    cropped = mask_img.crop(bbox)
    cw, ch = cropped.size
    # Compute scale to fit within margin
    scale = min((tw * (1 - margin)) / cw, (th * (1 - margin)) / ch)
    ns = (max(1, int(cw * scale)), max(1, int(ch * scale)))
    scaled = cropped.resize(ns, Image.LANCZOS)
    px = (tw - ns[0]) // 2
    py = (th - ns[1]) // 2
    # Paste silhouette as black fill
    debug.paste((0, 0, 0), box=(px, py, px + ns[0], py + ns[1]), mask=scaled)
    # Draw a simple outline around the silhouette by drawing a rectangle around its bounding area
    draw = ImageDraw.Draw(debug)
    draw.rectangle([px - 1, py - 1, px + ns[0], py + ns[1]], outline=(30, 30, 30), width=1)
    debug.save(outfile)

# ------------------------------------------------------------------
# Compositing texture into mask silhouette
# ------------------------------------------------------------------
def apply_texture_to_mask(texture_img, mask_img, bg_color=(6, 6, 8)):
    """Composite a texture into a mask silhouette.

    A new RGB image filled with bg_color is created. The texture
    image is pasted on top using the mask image as the alpha channel.
    """
    out = Image.new("RGB", texture_img.size, bg_color)
    out.paste(texture_img, (0, 0), mask_img)
    return out

# ------------------------------------------------------------------
# Facial feature drawing
# ------------------------------------------------------------------
def _lighten(color, factor=0.5):
    """Lighten an RGB color by blending with white.

    factor of 0 gives the original color; factor of 1 gives pure white.
    """
    return tuple(int((1 - factor) * c + factor * 255) for c in color)

def _darken(color, factor=0.5):
    """Darken an RGB color by blending with black.

    factor of 0 gives the original color; factor of 1 gives pure black.
    """
    return tuple(int((1 - factor) * c) for c in color)

def draw_hair(img, mask_img, palette, seed, sex_hint='unknown'):
    """Draw a stylized hair shape on top of the face silhouette.

    Hair is drawn above the top of the head using a polygon whose
    height and waviness are determined by a random generator seeded by
    `seed`. Male hair tends to be shorter; female hair is longer and
    frames the sides. The hair color is a darkened version of one of
    the palette colors. The hair polygon is clipped to the image
    dimensions (it is drawn directly on the image without masking).
    """
    bbox = mask_img.getbbox()
    if not bbox:
        return
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    cx = (x0 + x1) / 2.0
    # Determine hair height ratio based on sex
    # Create a deterministic random generator; mix seed with the word 'hair'
    rng = random.Random(hash((seed, 'hair')))
    if sex_hint == 'male':
        # Males tend to have shorter hair; keep ratio moderate
        hair_ratio = rng.uniform(0.15, 0.25)
    elif sex_hint == 'female':
        # Females tend to have longer hair; allow a broader range but avoid extreme heights
        hair_ratio = rng.uniform(0.20, 0.35)
    else:
        # Unknown: somewhere in between
        hair_ratio = rng.uniform(0.15, 0.30)
    hair_height = h * hair_ratio
    # Compute top-of-head y coordinate (approx the lowest y of top arc) by scanning mask
    # Use y0 as approximate top of head
    top_y = y0
    # Construct a noisy hair outline that conforms to the top of the head. We sample
    # points across the width of the head, find the silhouette's top at each
    # position, then extend upward by a sex‑dependent amount to create the hairline.
    draw = ImageDraw.Draw(img)
    # Base hair color is a darkened palette hue to contrast with the facial texture
    hair_color = _darken(palette[3], factor=0.6)
    # Determine the number of samples across the head width to capture hairline detail
    segments = max(30, int(w * 0.15))
    hair_top_points = []
    head_top_points = []
    for i in range(segments + 1):
        t = i / segments
        x = x0 + t * w
        col_x = int(x)
        col_x = min(max(col_x, 0), mask_img.width - 1)
        # Scan downward to find the top of the head (first non‑zero mask pixel)
        head_top_y = None
        for yy in range(max(0, int(y0) - int(hair_height * 1.2)), min(mask_img.height, int(y1))):
            if mask_img.getpixel((col_x, yy)) > 0:
                head_top_y = yy
                break
        if head_top_y is None:
            continue
        # Random hair length modifier by sex yields varied hairline heights
        if sex_hint == 'male':
            # Shorter strand length variation for males
            var = rng.uniform(0.6, 1.2)
        elif sex_hint == 'female':
            # Longer variation for females, but trimmed to avoid extremely long hair
            var = rng.uniform(1.0, 1.8)
        else:
            var = rng.uniform(0.8, 1.5)
        hair_len = hair_height * var
        hair_top_y = head_top_y - hair_len
        hair_top_points.append((x, hair_top_y))
        head_top_points.append((x, head_top_y))
    # If we gathered points successfully, fill the hair shape
    if hair_top_points and head_top_points:
        poly = hair_top_points + head_top_points[::-1]
        # Clip coordinates to image bounds
        poly_clipped = [(max(0, min(px, img.width - 1)), max(0, min(py, img.height - 1))) for (px, py) in poly]
        # Fill the hair area with the base color
        draw.polygon(poly_clipped, fill=hair_color)
        # Add wavy hair strands for a more organic look. Each strand is a polyline
        # starting at the hairline and ending at the head, with slight horizontal
        # jitter to mimic natural strands.
        stroke_count = max(10, int(w * 0.12))
        for j in range(stroke_count):
            tt = (j + rng.random()) / stroke_count
            # Base x position for this strand
            sx = x0 + tt * w
            col_x2 = int(min(max(int(sx), 0), mask_img.width - 1))
            # Find the head top again for this x
            head_y = None
            for yy in range(max(0, int(y0) - int(hair_height * 1.2)), min(mask_img.height, int(y1))):
                if mask_img.getpixel((col_x2, yy)) > 0:
                    head_y = yy
                    break
            if head_y is None:
                continue
            # Approximate hair top y by interpolating hair_top_points
            idx = int(tt * (len(hair_top_points) - 1))
            ht_x, ht_y = hair_top_points[idx]
            # Construct a multi‑segment path with jittered x offsets
            path = []
            strand_len = head_y - ht_y
            segments_in_strand = 4
            for s in range(segments_in_strand + 1):
                frac = s / segments_in_strand
                y_pos = ht_y + frac * strand_len
                # small horizontal offset amplitude relative to head width
                offset_amp = w * 0.02
                x_pos = sx + rng.uniform(-offset_amp, offset_amp)
                # Clamp within image bounds
                x_pos_clamped = max(0, min(img.width - 1, x_pos))
                path.append((x_pos_clamped, y_pos))
            # Use a slightly lighter or darker color for strands
            strand_color = _darken(hair_color, factor=0.3 + rng.uniform(-0.1, 0.1))
            line_width = max(1, int(w * 0.002))
            draw.line(path, fill=strand_color, width=line_width)

def draw_facial_features(img, mask_img, palette, sex_hint='unknown', seed=None):
    """Draw simple stylized eyes, nose, and lips on the image inside the mask.

    The positions of the features are determined relative to the bounding box
    of the mask. Colors are derived from the provided palette and lightened
    for contrast against the dark texture. Features are drawn only within
    the silhouette region because the background is preserved outside the mask.
    """
    bbox = mask_img.getbbox()
    if not bbox:
        return
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    cx = (x0 + x1) / 2.0
    # Initialize a deterministic random generator if a seed is provided
    rng = random.Random(seed) if seed is not None else None
    # Eye parameters (position and size scaled to face)
    eye_y = y0 + h * 0.35
    eye_offset_x = w * 0.20
    eye_w = w * 0.12
    eye_h = h * 0.06
    # Optional jitter for more hand‑drawn style
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
    # Colors for features
    sclera_color = _lighten(palette[0], factor=0.7)
    iris_color = _lighten(palette[3], factor=0.3)
    pupil_color = (20, 20, 20)
    # Draw eyeballs (sclera)
    draw.ellipse(left_eye, fill=sclera_color)
    draw.ellipse(right_eye, fill=sclera_color)
    # Compute iris dimensions
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
    # Draw pupils at the center of the iris
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
    # Draw highlights on pupils for a lively look
    highlight_scale = 0.35
    hl_w = pupil_w * highlight_scale
    hl_h = pupil_h * highlight_scale
    # Left eye highlight offset to the upper right
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
    # Outline the eyes to suggest eyelids and eyelashes
    outline_color = (30, 30, 30)
    outline_width = max(1, int(w * 0.003))
    draw.ellipse(left_eye, outline=outline_color, width=outline_width)
    draw.ellipse(right_eye, outline=outline_color, width=outline_width)
    # Add eyelid arcs (top lids) to give the eyes more character
    lid_width = max(1, int(outline_width * 1.5))
    draw.arc(left_eye, start=180, end=360, fill=outline_color, width=lid_width)
    draw.arc(right_eye, start=180, end=360, fill=outline_color, width=lid_width)
    # Nose: draw a triangular nose for a more human profile
    nose_color = _lighten(palette[2], factor=0.4)
    nose_top_y = y0 + h * 0.46
    nose_bottom_y = y0 + h * 0.64
    nose_bottom_w = w * 0.12
    if rng:
        # Apply slight randomness to the nose height and width
        nose_bottom_y = nose_bottom_y + h * 0.02 * (rng.random() - 0.5)
        nose_bottom_w = nose_bottom_w + w * 0.015 * (rng.random() - 0.5)
    nose_apex = (cx, nose_top_y)
    nose_left = (cx - nose_bottom_w / 2, nose_bottom_y)
    nose_right = (cx + nose_bottom_w / 2, nose_bottom_y)
    draw.polygon([nose_apex, nose_right, nose_left], fill=nose_color)
    # Draw nostrils as small dark ovals near the base of the nose
    nostril_color = (30, 30, 30)
    nostril_radius_x = w * 0.015
    nostril_radius_y = h * 0.01
    nostril_y = nose_bottom_y - nostril_radius_y * 0.5
    # Left nostril
    draw.ellipse([
        cx - nose_bottom_w * 0.25 - nostril_radius_x,
        nostril_y - nostril_radius_y,
        cx - nose_bottom_w * 0.25 + nostril_radius_x,
        nostril_y + nostril_radius_y], fill=nostril_color)
    # Right nostril
    draw.ellipse([
        cx + nose_bottom_w * 0.25 - nostril_radius_x,
        nostril_y - nostril_radius_y,
        cx + nose_bottom_w * 0.25 + nostril_radius_x,
        nostril_y + nostril_radius_y], fill=nostril_color)
    # Lips: draw upper and lower lips as smooth ellipses with a dividing line
    # Lips: use a lighter shade for better contrast
    lip_color = _lighten(palette[1], factor=0.6)
    lip_y_center = y0 + h * 0.77
    lip_w = w * 0.32
    lip_h_total = h * 0.075
    if rng:
        lip_w = lip_w + w * 0.02 * (rng.random() - 0.5)
        lip_h_total = lip_h_total + h * 0.02 * (rng.random() - 0.5)
    top_lip_h = lip_h_total * 0.45
    top_lip_box = [cx - lip_w / 2, lip_y_center - lip_h_total / 2,
                   cx + lip_w / 2, lip_y_center - lip_h_total / 2 + top_lip_h]
    bottom_lip_h = lip_h_total * 0.55
    bottom_lip_box = [cx - lip_w / 2, lip_y_center - lip_h_total / 2 + top_lip_h,
                      cx + lip_w / 2, lip_y_center + lip_h_total / 2]
    draw.ellipse(top_lip_box, fill=lip_color)
    draw.ellipse(bottom_lip_box, fill=lip_color)
    # Draw a slightly darker dividing line between lips
    lip_line_color = (50, 50, 50)
    line_y = lip_y_center - lip_h_total / 2 + top_lip_h
    draw.line([cx - lip_w / 2, line_y, cx + lip_w / 2, line_y], fill=lip_line_color, width=max(1, int(w * 0.015)))


# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python genome_flow_face_fixed.py /path/to/23andme.txt output_prefix")
        sys.exit(1)
    infile, outpref = sys.argv[1], sys.argv[2]
    if not os.path.isfile(infile):
        print("File not found:", infile)
        sys.exit(1)
    rows = parse_23andme_with_chr(infile)
    if not rows:
        print("No SNP rows parsed. Check the input file format.")
        sys.exit(1)
    # Flatten genotypes and compute seed & palette
    genotypes = [gt for _, gt in rows]
    gen_bytes = genotypes_to_bytes(genotypes)
    h = hash_bytes(gen_bytes)
    seed = int.from_bytes(h[:8], 'big')
    palette = palette_from_hash(h, n=6)
    sex_hint = sex_from_rows(rows)
    print("Inferred sex:", sex_hint)
    size = (1024, 1024)
    # Generate flow texture
    texture = flow_field_texture(size, palette, seed, num_lines=1800, steps=120, bg=(0, 0, 0))
    texture.save(f"{outpref}_flow_var.png")
    # Build masks
    mask_real = build_face_realistic_mask(size, seed, sex_hint)
    mask_styl = build_face_stylized_mask(size, seed + 1, sex_hint)
    # Export debug masks
    export_debug_mask(mask_real, f"{outpref}_mask_realistic.png", target_size=size)
    export_debug_mask(mask_styl, f"{outpref}_mask_stylized.png", target_size=size)
    # Composite texture into masks with a light background to reveal silhouette edges
    face_real = apply_texture_to_mask(texture, mask_real, bg_color=(245, 245, 248))
    face_styl = apply_texture_to_mask(texture, mask_styl, bg_color=(245, 245, 248))
    # Draw facial features (eyes, nose, lips) on each face
    draw_facial_features(face_real, mask_real, palette, sex_hint, seed=seed)
    draw_facial_features(face_styl, mask_styl, palette, sex_hint, seed=seed+1)
    # Draw hair on top of the faces for added individuality
    draw_hair(face_real, mask_real, palette, seed, sex_hint)
    draw_hair(face_styl, mask_styl, palette, seed+2, sex_hint)
    # Save the final face images
    face_real.save(f"{outpref}_face_realistic.png")
    face_styl.save(f"{outpref}_face_stylized.png")
    print("Done generating files with prefix", outpref)

if __name__ == "__main__":
    main()