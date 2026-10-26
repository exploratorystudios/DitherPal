"""A textured neural-depth surface, shared by every 3D animation dither mode."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .artifacts import write_viewer
from .geometry import build_mesh, write_obj, write_ply
from . import layers as hidden
from .model import scale_depth
from .renderer import rasterize
from .repairs import DonorRepair

DETAIL_GRID, MAX_DETAIL = 1024, 4   # Geometry vertices along the longer side / per source pixel.
START_METHOD = 'fork'               # Workers share the parent's scene memory copy-on-write.
FRAME_TIMEOUT = 300                 # Seconds to wait for one parallel camera angle.


class NeuralScene:
    def __init__(self, image, near, strength=100, resolution=720):
        self.strength = strength
        image = image.convert('RGB')
        image.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
        image = image.resize((max(2, image.width//2*2), max(2, image.height//2*2)), Image.Resampling.LANCZOS)
        self.rgb = np.array(image)
        self.texture = self.rgb
        self.repairs = None
        self.layers = []
        self.source_near = cv2.resize(np.asarray(near, np.float32), image.size, interpolation=cv2.INTER_LINEAR)
        source = cv2.resize(scale_depth(near, strength), image.size, interpolation=cv2.INTER_LINEAR)
        self.background_z = float(1/(.25+.75*hidden.subject_mask(source)[1]))
        # A dense grid lets tears follow depth edges closely and lets soft slopes, such as a
        # hand against an arm, stretch instead of tearing open.
        detail = min(MAX_DETAIL, DETAIL_GRID/max(image.size))
        self.geometry_size = tuple(max(2, round(n*detail)) for n in image.size)
        self.near = self._geometry_near(source)
        self.z = (1/(.25+.75*self.near)).astype(np.float32)
        self.mesh = build_mesh(self.z, mesh_size=max(self.geometry_size))
        if not len(self.mesh.faces):
            raise ValueError('Neural depth produced no usable triangles')
        # Background donor patches follow the coarse source-resolution tears. Hidden layers
        # cover most gaps, and per-patch pixel-art shading grows with the number of edges.
        self.repair_z = (1/(.25+.75*source)).astype(np.float32)
        self.repair_mesh = build_mesh(self.repair_z, mesh_size=min(256, max(image.size)))

    def _geometry_near(self, near):
        if near.shape[::-1] == self.geometry_size:
            return near
        return np.clip(cv2.resize(near, self.geometry_size, interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)

    def prepare_repairs(self, cancelled=lambda: False, progress=None, pixel_art=None):
        if self.repairs is None:
            self.repairs = DonorRepair(self.repair_mesh, self.texture, self.repair_z, cancelled, progress, pixel_art)

    def prepare_layers(self, inpaint, bake=None, cancelled=lambda: False, progress=None):
        """Paint hidden layers behind depth steps. inpaint(rgb, mask) paints source pixels;
        bake(painted_rgb, near) styles them like the main texture (default: resize only)."""
        if self.layers:
            return
        height, width = self.texture.shape[:2]
        prepared = []
        for name, mask in hidden.layer_masks(self.source_near).items():
            if cancelled():
                raise InterruptedError()
            if progress:
                progress(f'Painting the hidden {name} layer…')
            near = scale_depth(hidden.layer_near(self.source_near, mask), self.strength)
            layer = hidden.HiddenLayer(name, mask, near, inpaint(self.rgb, mask))
            z = (1/(.25+.75*self._geometry_near(near))).astype(np.float32)
            layer.mesh = hidden.sheet(z, mask)
            styled = (bake(layer.painted, near) if bake is not None else
                      cv2.resize(layer.painted, (width, height), interpolation=cv2.INTER_NEAREST))
            inside = cv2.resize(np.uint8(mask), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            layer.texture = self.texture.copy()
            layer.texture[inside] = np.asarray(styled)[inside]
            prepared.append(layer)
        if cancelled():
            raise InterruptedError()
        # Nearest layer first: the inner layer sits in front of the sky layer.
        self.layers = sorted(prepared, key=lambda layer: layer.name != 'inner')

    def write(self, folder, label, dolly):
        folder = Path(folder)
        Image.fromarray(self.texture).save(folder/'texture.png')
        np.save(folder/'camera_z.npy', self.z)
        Image.fromarray(np.uint16(np.round(self.near*65535))).save(folder/'depth_16bit.png')
        np.savez_compressed(folder/'scene.npz', vertices=self.mesh.vertices, uv=self.mesh.uv,
                            faces=self.mesh.faces, width=self.mesh.width, height=self.mesh.height, focal=self.mesh.focal)
        write_obj(folder, self.mesh)
        write_ply(folder/'scene.ply', self.mesh, self.texture)
        if self.repairs is not None:
            self.repairs.write(folder)
        for layer in self.layers:
            Image.fromarray(np.uint8(layer.mask)*255).save(folder/f'hidden_{layer.name}_mask.png')
            Image.fromarray(layer.painted).save(folder/f'hidden_{layer.name}_painted.png')
            Image.fromarray(layer.texture).save(folder/f'hidden_{layer.name}_texture.png')
        viewer = folder/'viewer.html'
        write_viewer(viewer, self.mesh, self.texture, label, dolly)
        return str(viewer.resolve())

    def frames(self, folder, count=48, dolly=.14, sway=.02, fill_holes=True, cancelled=lambda: False,
               output_size=None, cell=1, workers=1):
        """cell is the art-pixel size in output pixels, for seams drawn in whole art pixels.
        workers > 1 renders camera angles in parallel processes; frames still arrive in order."""
        width, height = output_size or (self.mesh.width, self.mesh.height)
        if fill_holes:
            self.prepare_repairs(cancelled)
        layers = self.layers if fill_holes else []
        coverage = []
        donor_coverage = []
        layer_coverage = []
        jobs = [(index, count, dolly, sway, fill_holes, (width, height), cell, str(folder)) for index in range(count)]
        results = _parallel(self, jobs, workers, cancelled) if workers > 1 and count > 1 else (
            self._frame(*job) for job in jobs)
        for frame, near, covered, donors, shown in results:
            if cancelled():
                return
            coverage.append(covered)
            if fill_holes:
                donor_coverage.append(donors)
            if layers:
                layer_coverage.append(shown)
            yield frame, near
        report = dict(frames=count, minimum_coverage=min(coverage), dolly=dolly, sway=sway, depth_strength=self.strength,
                      fill_holes=fill_holes, vertices=len(self.mesh.vertices), triangles=len(self.mesh.faces),
                      render_size=[width, height], render_workers=workers if count > 1 else 1, texture_size=list(self.texture.shape[1::-1]), texture_filter='bilinear',
                      gap_repair=self.repairs.summary() if fill_holes else None,
                      donor_pixels_per_frame=donor_coverage,
                      hidden_layers=dict(method='LaMa inpainted sheets behind depth steps; smooth inner seams',
                                         layers={layer.name: dict(painted_pixels=int(layer.mask.sum()),
                                                                  triangles=len(layer.mesh.faces))
                                                 for layer in layers},
                                         pixels_per_frame=layer_coverage) if layers else None,
                      limitation='Single-photo visible surface. Gap filling does not reconstruct hidden geometry.')
        (Path(folder)/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def _frame(self, index, count, dolly, sway, fill_holes, size, cell, folder):
        """One camera angle: raw surface, per-angle healed repairs, hidden layers."""
        width, height = size
        focal = self.mesh.focal * width / self.mesh.width
        layers = self.layers if fill_holes else []
        phase = 2*np.pi*index/count
        camera = np.array((sway*np.sin(phase), 0, dolly*(1-np.cos(phase))/2), np.float32)
        raw, z = rasterize(self.mesh.vertices, self.mesh.uv, self.mesh.faces, self.texture,
                           width, height, focal, camera, 0.)
        missing = ~np.isfinite(z)
        frame, donors, layer_pixels = raw, None, None
        if fill_holes:
            frame, repaired_z, patch_alpha, feather = self.repairs.render(raw, z, focal, camera)
            if layers:
                frame, repaired_z, shown = self._composite(raw, z, frame, repaired_z, layers, width, height,
                                                           focal, camera, cell)
                layer_pixels = int((missing & shown).sum())
            near = np.clip((1/repaired_z-.25)/.75, 0, 1).astype(np.float32)
            remaining = ~np.isfinite(repaired_z)
            if remaining.any() and not remaining.all():
                near = np.clip(cv2.inpaint(near, np.uint8(remaining)*255, 3, cv2.INPAINT_TELEA), 0, 1)
            donors = int((missing & (patch_alpha > .5)).sum())
        else:
            near = np.clip((1/z-.25)/.75, 0, 1).astype(np.float32)
        if index == count//2:
            Image.fromarray(raw).save(Path(folder)/'zoom_peak_raw.png')
            Image.fromarray(np.uint8(missing)*255).save(Path(folder)/'uncovered_mask.png')
            if fill_holes:
                # This angle's own atlases, so the parallel worker that renders the peak writes them.
                self.repairs.write_peak(folder)
                if layers:
                    Image.fromarray(np.uint8(missing & shown)*255).save(Path(folder)/'hidden_layer_mask.png')
                Image.fromarray(np.uint8(missing & (patch_alpha > .5))*255).save(Path(folder)/'donor_patch_mask.png')
                Image.fromarray(np.uint8(np.round(feather*255))).save(Path(folder)/'foreground_feather_alpha.png')
        return frame, near, float(1-missing.mean()), donors, layer_pixels

    def _composite(self, raw, z, repaired, repaired_z, layers, width, height, focal, camera, cell):
        """Gaps show the nearest hidden layer; any gap no layer covers gets the smooth inner
        seam over the background repair."""
        scale = width/self.rgb.shape[1]
        result = hidden.smooth_seams(raw, z, repaired, cell, self.background_z, scale)
        missing = ~np.isfinite(z)
        best = np.full(z.shape, np.inf, np.float32)
        for layer in layers:
            colour, depth = rasterize(layer.mesh.vertices, layer.mesh.uv, layer.mesh.faces, layer.texture,
                                      width, height, focal, camera, 0.)
            take = missing & (depth*hidden.LAYER_PUSH < best)
            result[take], best[take] = colour[take], depth[take]*hidden.LAYER_PUSH
        shown = np.isfinite(best)
        return result, np.where(shown, best, repaired_z), shown


_worker_scene = None


def _render_job(job):
    return _worker_scene._frame(*job)


def _parallel(scene, jobs, workers, cancelled):
    """Camera angles are independent, so their per-angle healing runs side by side.
    If a worker dies (for example under memory pressure), the remaining angles render serially."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    global _worker_scene
    # A forked worker inherits this reference: the mesh, textures and repair patches are
    # shared, never pickled (they are close to 1 GB for a dense pixel-art scene).
    # Do not call cv2.setNumThreads in a forked worker: it waits on the parent's pool threads forever.
    _worker_scene = scene
    done = 0
    executor = ProcessPoolExecutor(min(workers, len(jobs)), multiprocessing.get_context(START_METHOD))
    try:
        futures = [executor.submit(_render_job, job) for job in jobs]
        for future in futures:
            # A stuck worker must not stall the render: stop waiting and finish serially.
            result = future.result(timeout=FRAME_TIMEOUT)
            if cancelled():
                return
            done += 1
            yield result
    except (BrokenProcessPool, TimeoutError):
        _kill_workers(executor)
        for job in jobs[done:]:
            if cancelled():
                return
            yield scene._frame(*job)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        _kill_workers(executor)
        _worker_scene = None


def _kill_workers(executor):
    if hasattr(executor, 'kill_workers'):   # Python 3.14+
        executor.kill_workers()
        return
    for process in list((getattr(executor, '_processes', None) or {}).values()):
        process.kill()
