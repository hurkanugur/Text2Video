import trimesh
import imageio
import numpy as np
import os

def render_mesh_sequence(mesh_template_path, params_seq, out_path="outputs/video.mp4"):
    mesh = trimesh.load(mesh_template_path)
    writer = imageio.get_writer(out_path, fps=25)

    for p in params_seq:
        # Fake: apply expression offsets as vertex noise
        vertices = mesh.vertices + 0.01 * np.random.randn(*mesh.vertices.shape)
        mesh.vertices = vertices
        img = mesh.scene().save_image(resolution=(256, 256))
        writer.append_data(imageio.imread(img))

    writer.close()
