"""Selected pixel-art shading on isolated background repair materials."""
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SeamPixelArt:
    style: str
    art_size: tuple
    art_block: int
    cell: int
    grid_shape: tuple
    colors: int
    strength: float
    preserve_highlights: bool
    quantize: object
    pattern: object

    def shade(self, image, near, palette):
        art = self.quantize(image, self.cell, self.colors, 0., palette,
                            preserve_highlights=self.preserve_highlights)
        if self.style == 'high_res':
            return self.pattern(image, art, self.cell, near, self.strength, extend_palette=False)
        if self.style == 'retro':
            return self.pattern(art, self.cell, near, self.strength)
        raise ValueError('Seam pixel art requires High Res or Retro')

    def prepare(self, mesh, texture, depth, metadata, cancelled=lambda: False):
        dims = np.array([mesh.width-1, mesh.height-1], np.float32)
        for item in metadata:
            if cancelled():
                raise InterruptedError()
            xy = item['world_xy']*mesh.focal/item['z'][..., None]+dims/2
            cols = np.floor((xy[..., 0]+.5)*self.art_size[0]/mesh.width/self.art_block).astype(int)
            rows = np.floor((xy[..., 1]+.5)*self.art_size[1]/mesh.height/self.art_block).astype(int)
            item['grid_columns'] = np.clip(cols, 0, self.grid_shape[1]-1)
            item['grid_rows'] = np.clip(rows, 0, self.grid_shape[0]-1)
            item['art_context'] = self._context(item, mesh, texture, depth)

    def _context(self, item, mesh, texture, depth):
        p = item['patch']
        dims = np.array([mesh.width-1, mesh.height-1], np.float32)
        texture_scale = np.array([(texture.shape[1]-1)/dims[0], (texture.shape[0]-1)/dims[1]])
        xy = item['world_xy']*mesh.focal/item['z'][..., None]+dims/2
        donor = xy-p['normal']*p['offset']+p['tangent']*p['shift']
        dx, dy = np.rint(donor[..., 0]).astype(int), np.rint(donor[..., 1]).astype(int)
        inside = (dx >= 0) & (dx < mesh.width) & (dy >= 0) & (dy < mesh.height)
        dx, dy = np.clip(dx, 0, mesh.width-1), np.clip(dy, 0, mesh.height-1)
        donor_z = depth[dy, dx]
        safe = inside & np.isfinite(donor_z)
        safe &= np.maximum(donor_z/item['z'], item['z']/donor_z) < 1.15
        if not safe.any():
            return None
        tx = np.clip(np.rint(donor[..., 0]*texture_scale[0]).astype(int), 0, texture.shape[1]-1)
        ty = np.clip(np.rint(donor[..., 1]*texture_scale[1]).astype(int), 0, texture.shape[0]-1)
        palette = np.unique(texture[ty[safe], tx[safe]], axis=0)
        if 'allowed_background_palette' in p:
            palette = p['allowed_background_palette']
        cols, rows = item['grid_columns'], item['grid_rows']
        col0, row0 = (int(cols.min())-4)//4*4, (int(rows.min())-4)//4*4
        col1, row1 = (int(cols.max())+8)//4*4, (int(rows.max())+8)//4*4
        gw, gh = col1-col0, row1-row0
        x = (col0*self.art_block+(np.arange(gw*self.cell)+.5)*self.art_block/self.cell)*mesh.width/self.art_size[0]-.5
        y = (row0*self.art_block+(np.arange(gh*self.cell)+.5)*self.art_block/self.cell)*mesh.height/self.art_size[1]-.5
        xx, yy = np.meshgrid(x, y)
        sx, sy = np.rint(xx).astype(int), np.rint(yy).astype(int)
        inside = (sx >= 0) & (sx < mesh.width) & (sy >= 0) & (sy < mesh.height)
        sx, sy = np.clip(sx, 0, mesh.width-1), np.clip(sy, 0, mesh.height-1)
        tx = np.clip(np.rint(xx*texture_scale[0]).astype(int), 0, texture.shape[1]-1)
        ty = np.clip(np.rint(yy*texture_scale[1]).astype(int), 0, texture.shape[0]-1)
        image, z = texture[ty, tx].copy(), depth[sy, sx].copy()
        delta = np.stack([xx-p['points'][0, 0], yy-p['points'][0, 1]], axis=-1)
        along, across = delta@p['tangent'], delta@p['normal']
        expected = np.interp(along, [0, p['length']], p['depth'])
        valid = inside & np.isfinite(z) & (np.maximum(z/expected, expected/z) < 1.15)
        belongs = np.zeros(valid.shape, bool)
        for color in palette:
            belongs |= (image == color).all(axis=2)
        valid &= belongs
        if valid.any():
            _, labels = cv2.distanceTransformWithLabels(np.uint8(~valid), cv2.DIST_L2, 5,
                                                        labelType=cv2.DIST_LABEL_PIXEL)
            seeds = np.flatnonzero(valid)
            nearest = seeds[np.clip(labels-1, 0, len(seeds)-1)]
            image, z = image.reshape(-1, 3)[nearest].copy(), z.ravel()[nearest].copy()
        else:
            image[:] = palette[0]
            z[:] = expected
        extension = p.get('extension', 12.)
        region = (along >= -1) & (along <= p['length']+1) & (across >= -3) & (across <= extension)
        h, w = p['copied'].shape[:2]
        px = np.ascontiguousarray((along+1)/(p['length']+2)*(w-1), np.float32)
        py = np.ascontiguousarray((across+3)/(extension+3)*(h-1), np.float32)
        z[region] = expected[region]
        near = np.clip((1/z-.25)/.75, 0, 1).astype(np.float32)
        return dict(image=image, near=near, region=region, px=px, py=py, palette=palette,
                    rows=rows-row0, cols=cols-col0, cache=OrderedDict())

    def dither(self, atlas, metadata, cancelled=lambda: False):
        result = atlas.copy()
        for item in metadata:
            if cancelled():
                raise InterruptedError()
            context = item['art_context']
            if context is None:
                continue
            x, y = item['slot']
            h, w = item['patch']['copied'].shape[:2]
            source = atlas[y:y+h, x:x+w]
            key, cache = source.tobytes(), context['cache']
            if key in cache:
                quantized = cache[key]
                cache.move_to_end(key)
            else:
                image = context['image'].copy()
                copied = cv2.remap(source, context['px'], context['py'], cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                image[context['region']] = copied[context['region']]
                art = self.shade(image, context['near'], context['palette'])
                grid = art[::self.cell, ::self.cell]
                quantized = grid[context['rows'], context['cols']]
                cache[key] = quantized
                if len(cache) > 8:
                    cache.popitem(last=False)
            result[y-2:y+h+2, x-2:x+w+2] = cv2.copyMakeBorder(quantized, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
        return result
