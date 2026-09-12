"""Raw particle rendering without modifying the physical radii."""
import numpy as np
from PIL import Image, ImageDraw

def raw_particles(path, size=512, supersample=4):
    """Actual disk radii/centers; supersampling integrates subpixel coverage.

    No dilation, hole filling, synthetic grains, or Gaussian smoothing.
    Rasterize at high resolution and reduce ONCE to the requested size.
    """
    s = np.load(path)
    high = size*supersample
    canvas = Image.new('L', (high, high), 255)
    draw = ImageDraw.Draw(canvas)
    scale = high/.5
    for x, radius, captured in zip(s['x'], s['r'], s['captured']):
        if captured:
            continue
        px = (x[0]-.1875)*scale; py = (.6875-x[1])*scale
        r = radius*scale
        draw.ellipse((px-r, py-r, px+r, py+r), fill=0)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def assemble(ims, keys, columns):
    canvas = Image.new('L', (256*columns, 256*((len(keys)+columns-1)//columns)), 255)
    for i, key in enumerate(keys):
        canvas.paste(ims[key].resize((256, 256), Image.Resampling.LANCZOS),
                     (256*(i % columns), 256*(i//columns)))
    return canvas


