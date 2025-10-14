import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import math
import random

def palette_from_emotion(valence, arousal):
    # Normalize to [0,1]
    v = max(0.0, min(10.0, float(valence))) / 10.0
    a = max(0.0, min(10.0, float(arousal))) / 10.0
    # Hue by valence, saturation by arousal
    base_h = int(360 * v)
    sat = 50 + int(50 * a)
    bri = 50 + int(50 * (0.5 + 0.5 * a))
    # Return 3 complementary hues
    hues = [(base_h + d) % 360 for d in (0, 120, 240)]
    colors = [hsv_to_rgb(h/360.0, sat/100.0, bri/100.0) for h in hues]
    return [(int(r*255), int(g*255), int(b*255)) for r,g,b in colors]

def hsv_to_rgb(h, s, v):
    i = int(h*6); f = h*6 - i
    p = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s)
    i = i % 6
    if i == 0: r,g,b = v,t,p
    elif i == 1: r,g,b = q,v,p
    elif i == 2: r,g,b = p,v,t
    elif i == 3: r,g,b = p,q,v
    elif i == 4: r,g,b = t,p,v
    else:        r,g,b = v,p,q
    return r,g,b

def render_abstract(valence, arousal, size=128, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    colors = palette_from_emotion(valence, arousal)
    num_layers = 6 + int(6 * (arousal / 10.0))    # busier with higher arousal
    radius_base = 8 + int(30 * (valence / 10.0))  # softer/rounder with higher valence

    for i in range(num_layers):
        c = colors[i % len(colors)]
        alpha = 40 + int(140 * (valence / 10.0))
        r = radius_base + np.random.randint(0, 20)
        x = np.random.randint(r, size - r)
        y = np.random.randint(r, size - r)
        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill=(c[0], c[1], c[2], alpha))

        # occasional strokes (more energetic with arousal)
        if np.random.rand() < (arousal / 10.0):
            x2 = np.random.randint(0, size)
            y2 = np.random.randint(0, size)
            draw.line([x, y, x2, y2],
                      fill=(c[0], c[1], c[2], 80),
                      width=1 + int(3 * (arousal / 10.0)))
    if np.random.rand() < 0.9:  # apply most of the time
        sat_factor = 1.0 + np.random.uniform(-0.3, 0.6)
        bright_factor = 1.0 + np.random.uniform(-0.2, 0.4)
        contrast_factor = 1.0 + np.random.uniform(-0.2, 0.4)

        img = ImageEnhance.Color(img).enhance(sat_factor)
        img = ImageEnhance.Brightness(img).enhance(bright_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # occasional inversion / tone shift to expand color space
    if np.random.rand() < 0.05:
        img = ImageOps.posterize(img, bits=np.random.choice([4, 5, 6]))

    return np.asarray(img).astype(np.float32) / 255.0