"""Inpainted hidden layers behind depth steps, and smooth seams for gaps they leave.

A camera move opens gaps where nearer geometry tears away from what lies behind it. The
sky layer paints out the whole subject; the inner layer paints out each foreground piece
that stands on a step in front of the subject's own interior (an arm across a robe). Each
layer is a continuous sheet meshed just behind the surface it continues, so a revealed gap
shows a plausible hidden surface instead of the far background.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Mesh, build_mesh

RIM, REACH, STEP, GROW = 2, 4, .08, 8   # Source pixels / normalized near.
SKY_GROW = 3
SHEET_GRID, MARGIN = 512, 6
LAYER_PUSH = 1.004                       # Hidden layers sit just behind the surface they continue.
SPLIT_RADIUS, MAX_REACH = 6., 12.        # Source pixels.
MEDIAN, MIDLINE_BLUR = 1.75, 2.          # Source pixels / art cells.


@dataclass
class HiddenLayer:
    name: str
    mask: np.ndarray       # Source-resolution pixels that were painted.
    near: np.ndarray       # Depth-strength-scaled, at source resolution.
    painted: np.ndarray    # Source-resolution RGB.
    mesh: Mesh = None
    texture: np.ndarray = None


def subject_mask(near):
    """Pixels nearer than the Otsu split between subject and background."""
    threshold, _ = cv2.threshold(np.uint8(np.clip(near, 0, 1)*255), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    subject = near > threshold/255
    return cv2.morphologyEx(np.uint8(subject), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0, threshold/255


def layer_masks(near):
    """Source-resolution masks for the sky and inner layers, from unscaled depth."""
    subject = subject_mask(near)[0]
    if not subject.any() or subject.all():
        return {}
    interior = cv2.erode(np.uint8(subject), np.ones((2*RIM+1,)*2, np.uint8)) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*REACH+1,)*2)
    rear_level = cv2.erode(np.where(interior, near, 1.).astype(np.float32), kernel)
    # Steep steps only: the soft outline rim and gentle interior slopes do not count.
    front = subject & (near-rear_level > STEP)
    # A gap reveals what lies under the front piece beside its edge: mask that band.
    band = cv2.dilate(np.uint8(front), np.ones((2*GROW+1,)*2, np.uint8)) > 0
    masks = dict(sky=cv2.dilate(np.uint8(subject), np.ones((2*SKY_GROW+1,)*2, np.uint8)) > 0,
                 inner=band & subject & (near > rear_level+STEP/4))
    return {name: mask for name, mask in masks.items() if mask.any()}


def layer_near(near, mask):
    """Continue the surrounding depth under the mask; never pull a hidden surface forward."""
    filled = cv2.inpaint(np.uint16(np.round(np.clip(near, 0, 1)*65535)), np.uint8(mask), 6,
                         cv2.INPAINT_TELEA).astype(np.float32)/65535
    return np.where(mask, np.minimum(filled, near), near).astype(np.float32)


def sheet(z, mask):
    """An untorn mesh trimmed to the mask: a hidden layer has nothing behind it to reveal."""
    mesh = build_mesh(z, mesh_size=SHEET_GRID, max_depth_jump=1e6)
    keep = cv2.dilate(np.uint8(mask), np.ones((2*MARGIN+1,)*2, np.uint8)) > 0
    h, w = keep.shape
    cols = np.clip(np.rint(mesh.uv[:, 0]*(w-1)).astype(int), 0, w-1)
    rows = np.clip(np.rint(mesh.uv[:, 1]*(h-1)).astype(int), 0, h-1)
    faces = mesh.faces[keep[rows, cols][mesh.faces].any(axis=1)]
    return Mesh(mesh.vertices, mesh.uv, np.ascontiguousarray(faces), mesh.width, mesh.height, mesh.focal)


def nearest(mask):
    """Distance to, and flat index of, the nearest True pixel."""
    distance, labels = cv2.distanceTransformWithLabels(np.uint8(~mask), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
    rows, cols = np.nonzero(mask)
    lookup = np.zeros(labels.max()+1, np.int64)
    lookup[labels[rows, cols]] = rows*mask.shape[1]+cols
    return distance, lookup[labels]


def smooth_seams(raw, depth, repaired, cell, background_z, scale):
    """Fill gaps between two subject layers by splitting them between the front and rear colour.

    Gaps onto the far background keep the existing repair. Colours come from a median-filtered
    frame, so the dither checker is not stretched, and are drawn in whole art cells. scale is
    render pixels per source pixel.
    """
    holes = ~np.isfinite(depth)
    if not holes.any() or holes.all():
        return repaired
    finite = np.where(holes, np.nan, depth)
    kernel = np.ones((2*max(1, round(SPLIT_RADIUS*scale))+1,)*2, np.uint8)
    low = cv2.erode(np.where(holes, np.inf, depth).astype(np.float32), kernel)
    high = cv2.dilate(np.where(holes, -np.inf, depth).astype(np.float32), kernel)
    edge = ~holes & (cv2.dilate(np.uint8(holes), np.ones((3, 3), np.uint8)) > 0)
    front = edge & (finite < (low+high)/2)
    rear = edge & (finite >= (low+high)/2) & (finite < background_z)
    sky = edge & (finite >= background_z)
    if not front.any() or not rear.any():
        return repaired
    fd, fi = nearest(front)
    rd, ri = nearest(rear)
    bd = nearest(sky)[0] if sky.any() else np.full(holes.shape, np.inf, np.float32)
    height, width = holes.shape
    cy = np.minimum((np.arange(height)//cell)*cell+cell//2, height-1)
    cx = np.minimum((np.arange(width)//cell)*cell+cell//2, width-1)
    centre = np.ix_(cy, cx)
    # Inner only: the nearest rear subject pixel is closer than the far background.
    seam = holes[centre] & (fd[centre] <= MAX_REACH*scale) & (rd[centre] < bd[centre])
    if not seam.any():
        return repaired
    # Holes are black in the raw frame; give them their nearest visible colour first so
    # the median only sees real texture.
    visible = raw.reshape(-1, 3)[nearest(~holes)[1]]
    colours = cv2.medianBlur(visible, max(3, round(MEDIAN*scale)//2*2+1)).reshape(-1, 3)
    small = cv2.resize((fd/np.maximum(fd+rd, 1e-6)).astype(np.float32),
                       (max(1, width//cell), max(1, height//cell)), interpolation=cv2.INTER_AREA)
    ramp = cv2.resize(cv2.GaussianBlur(small, (0, 0), MIDLINE_BLUR), (width, height),
                      interpolation=cv2.INTER_NEAREST)[centre]
    chosen = np.where((ramp < .5)[..., None], colours[fi[centre]], colours[ri[centre]])
    result = repaired.copy()
    result[seam] = chosen[seam]
    return result
