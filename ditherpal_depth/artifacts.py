import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_viewer(path, mesh, texture, name, dolly):
    def encode(array, dtype):
        return base64.b64encode(array.astype(dtype).tobytes()).decode('ascii')

    image = io.BytesIO()
    Image.fromarray(texture).save(image, 'PNG')
    data = dict(name=name, vertices=encode(mesh.vertices, '<f4'), uv=encode(mesh.uv, '<f4'),
                faces=encode(mesh.faces, '<u4'), texture='data:image/png;base64,' + base64.b64encode(image.getvalue()).decode('ascii'),
                width=mesh.width, height=mesh.height, focal=mesh.focal, vertex_count=len(mesh.vertices),
                center_z=float(np.median(mesh.vertices[:, 2])), near_z=float(mesh.vertices[:, 2].min()),
                far_z=float(mesh.vertices[:, 2].max()), dolly=dolly)
    payload = json.dumps(data).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (Path(__file__).parent / 'viewer_template.html').read_text()
    Path(path).write_text(template.replace('__SCENE_DATA__', payload))


def contact_sheet(path, panels):
    tiles = []
    for title, rgb in panels:
        image = Image.fromarray(rgb)
        image.thumbnail((360, 300))
        tile = Image.new('RGB', (384, 344), '#111a26')
        tile.paste(image, ((384-image.width)//2, 36+(300-image.height)//2))
        ImageDraw.Draw(tile).text((12, 12), title, fill='white')
        tiles.append(tile)
    sheet = Image.new('RGB', (384*3, 344*((len(tiles)+2)//3)), '#111a26')
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index%3*384, index//3*344))
    sheet.save(path)
