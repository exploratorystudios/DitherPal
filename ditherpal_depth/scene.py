"""A textured neural-depth surface, shared by every 3D animation dither mode."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .artifacts import write_viewer
from .geometry import build_mesh, write_obj, write_ply
from .model import scale_depth
from .renderer import rasterize


class NeuralScene:
    def __init__(self, image, near, strength=100, resolution=720):
        self.strength = strength
        image = image.convert('RGB')
        image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
        image = image.resize((max(2, image.width//2*2), max(2, image.height//2*2)), Image.Resampling.LANCZOS)
        self.rgb = np.array(image)
        self.texture = self.rgb
        self.near = cv2.resize(scale_depth(near, strength), image.size, interpolation=cv2.INTER_LINEAR)
        self.z = (1/(.25+.75*self.near)).astype(np.float32)
        self.mesh = build_mesh(self.z, mesh_size=min(256, max(image.size)))
        if not len(self.mesh.faces):
            raise ValueError('Neural depth produced no usable triangles')

    def write(self, folder, label, dolly):
        folder = Path(folder)
        Image.fromarray(self.texture).save(folder/'texture.png')
        np.save(folder/'camera_z.npy', self.z)
        Image.fromarray(np.uint16(np.round(self.near*65535))).save(folder/'depth_16bit.png')
        np.savez_compressed(folder/'scene.npz', vertices=self.mesh.vertices, uv=self.mesh.uv,
                            faces=self.mesh.faces, width=self.mesh.width, height=self.mesh.height, focal=self.mesh.focal)
        write_obj(folder, self.mesh)
        write_ply(folder/'scene.ply', self.mesh, self.texture)
        viewer = folder/'viewer.html'
        write_viewer(viewer, self.mesh, self.texture, label, dolly)
        return str(viewer.resolve())

    def frames(self, folder, count=48, dolly=.14, sway=.02, fill_holes=True, cancelled=lambda: False,
               output_size=None):
        width, height = output_size or (self.mesh.width, self.mesh.height)
        focal = self.mesh.focal * width / self.mesh.width
        coverage = []
        for index in range(count):
            if cancelled():
                return
            phase = 2*np.pi*index/count
            camera = np.array((sway*np.sin(phase), 0, dolly*(1-np.cos(phase))/2), np.float32)
            raw, z = rasterize(self.mesh.vertices, self.mesh.uv, self.mesh.faces, self.texture,
                               width, height, focal, camera, 0.)
            missing = ~np.isfinite(z)
            near = np.clip((1/z-.25)/.75, 0, 1).astype(np.float32)
            frame = raw
            if fill_holes and missing.any() and not missing.all():
                mask = np.uint8(missing)*255
                frame = cv2.inpaint(raw, mask, 3, cv2.INPAINT_TELEA)
                near = np.clip(cv2.inpaint(near, mask, 3, cv2.INPAINT_TELEA), 0, 1)
            coverage.append(float(1-missing.mean()))
            if index == count//2:
                Image.fromarray(raw).save(Path(folder)/'zoom_peak_raw.png')
                Image.fromarray(np.uint8(missing)*255).save(Path(folder)/'uncovered_mask.png')
            yield frame, near
        report = dict(frames=count, minimum_coverage=min(coverage), dolly=dolly, sway=sway, depth_strength=self.strength,
                      fill_holes=fill_holes, vertices=len(self.mesh.vertices), triangles=len(self.mesh.faces),
                      render_size=[width, height], texture_size=list(self.texture.shape[1::-1]), texture_filter='bilinear',
                      limitation='Single-photo visible surface. Gap filling does not reconstruct hidden geometry.')
        (Path(folder)/'report.json').write_text(json.dumps(report, indent=2)+'\n')
