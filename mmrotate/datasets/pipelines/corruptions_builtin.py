# Copyright (c) OpenMMLab. All rights reserved.
"""Size-agnostic image corruptions (numpy-only core).

Re-implements the subset of Hendrycks' ``imagecorruptions`` that is needed for
the MVE, but works on arbitrary image sizes (the upstream ``fog`` is hardcoded
to a 256x256 plasma map and breaks on 1024x1024 remote-sensing tiles).

All functions take an HxWx3 ``uint8`` array and return an HxWx3 ``uint8`` array.
``fog`` and ``gaussian_noise`` are pure numpy; the optional blur corruptions
lazily import cv2.
"""
import numpy as np


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def plasma_fractal(mapsize=256, wibbledecay=3.0):
    """Diamond-square plasma fractal in [0, 1] (Hendrycks et al.)."""
    assert (mapsize & (mapsize - 1) == 0), 'mapsize must be a power of two'
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble,
                                                      array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, 1, axis=0)
        squareaccum += np.roll(squareaccum, 1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize_:stepsize,
                          stepsize // 2:mapsize_:stepsize]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize,
                 stepsize // 2:mapsize_:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize_:stepsize,
                 0:mapsize_:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def gaussian_noise(x, severity=1):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = x.astype(np.float32) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)
    return (x * 255).astype(np.uint8)


def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    x = x.astype(np.float32) / 255.
    max_val = x.max()
    h, w = x.shape[:2]
    mapsize = _next_pow2(max(h, w))
    fog_map = plasma_fractal(
        mapsize=mapsize, wibbledecay=c[1])[:h, :w][..., np.newaxis]
    x = x + c[0] * fog_map
    x = np.clip(x * max_val / (max_val + c[0]), 0, 1)
    return (x * 255).astype(np.uint8)


def brightness(x, severity=1):
    # simple additive brightness on the luminance (size-agnostic, no skimage)
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = x.astype(np.float32) / 255.
    x = np.clip(x + c, 0, 1)
    return (x * 255).astype(np.uint8)


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = x.astype(np.float32) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = np.clip((x - means) * c + means, 0, 1)
    return (x * 255).astype(np.uint8)


def _disk_kernel(radius, alias_blur=0.1):
    if radius <= 8:
        size = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        size = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    xx, yy = np.meshgrid(size, size)
    aliased = np.array((xx**2 + yy**2) <= radius**2, dtype=np.float32)
    import cv2
    aliased = cv2.GaussianBlur(aliased, ksize=ksize, sigmaX=alias_blur)
    return aliased / aliased.sum()


def defocus_blur(x, severity=1):
    import cv2
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = x.astype(np.float32) / 255.
    kernel = _disk_kernel(radius=c[0], alias_blur=c[1])
    channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
    x = np.clip(np.stack(channels, axis=-1), 0, 1)
    return (x * 255).astype(np.uint8)


def gaussian_blur(x, severity=1):
    import cv2
    c = [1, 2, 3, 4, 6][severity - 1]
    x = x.astype(np.float32) / 255.
    x = cv2.GaussianBlur(x, (0, 0), sigmaX=c)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


def spatter(x, severity=1):
    """Simplified, size-safe mud-spatter occlusion (non-physical corruption).

    Not a pixel-exact match to Hendrycks' spatter (which needs skimage +
    Canny + distanceTransform); this captures the essence -- random blurred
    liquid blobs occluding/tinting the image -- and works at any size. Used
    consistently for both training aug and evaluation.
    """
    import cv2
    thr = [0.58, 0.62, 0.66, 0.70, 0.74][severity - 1]
    x = x.astype(np.float32) / 255.
    h, w = x.shape[:2]
    liquid = np.random.normal(size=(h, w), loc=0.5, scale=0.3)
    liquid = cv2.GaussianBlur(liquid, (0, 0), sigmaX=3)
    mask = (liquid > thr).astype(np.float32)[..., None]
    mud = np.array([0.20, 0.30, 0.45], dtype=np.float32)  # dark brown blob
    out = x * (1 - mask) + mud * mask
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


CORRUPTIONS = {
    'gaussian_noise': gaussian_noise,
    'fog': fog,
    'brightness': brightness,
    'contrast': contrast,
    'defocus_blur': defocus_blur,
    'gaussian_blur': gaussian_blur,
    'spatter': spatter,
}
