"""Frame-by-frame geometric inspection from rear, side and rear-oblique views.
Read-only motion/model rendering; no simulation steps or trajectory edits.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, tempfile, hashlib
from pathlib import Path
import numpy as np, mujoco, imageio_ffmpeg
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
from view_motion import load, model_file
from plan_knee_entry import create_context


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=640, choices=[640, 960])
    ap.add_argument("--motion", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a new inspection directory")
    args.output.mkdir(parents=True)
    size = args.size
    scale = size / 640
    path, a = load(args.motion)
    p = create_context()
    heading = float(np.degrees(np.arctan2(p.R[1, 0], p.R[0, 0])))
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", round(19 * scale)
    )
    small = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", round(15 * scale)
    )
    N = len(a["timestamps"])
    fps = float(a["fps"])
    rows = []
    views = [
        ("rear", "正后方", heading, -13, 4.2),
        ("side", "左侧", heading - 90, -12, 3.3),
        ("rear_oblique", "右斜后方", heading + 45, -28, 3.35),
    ]
    pages = {v[0]: [] for v in views}
    with tempfile.TemporaryDirectory() as folder:
        m = mujoco.MjModel.from_xml_path(str(model_file(folder, a, True)))
        m.vis.global_.offwidth = size
        m.vis.global_.offheight = size
        m.vis.headlight.ambient[:] = 0.5
        d = mujoco.MjData(m)
        ids = [m.jnt_qposadr[m.joint(str(n)).id] for n in a["joint_names"]]
        renderer = mujoco.Renderer(m, size, size)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = p.world(p.edge - 0.05, 0, 1.10)
        video = imageio_ffmpeg.write_frames(
            str(args.output / "three_views.mp4"),
            (size * 3, size),
            fps=fps,
            codec="libx264",
            quality=8,
            pix_fmt_out="yuv420p",
        )
        rear = imageio_ffmpeg.write_frames(
            str(args.output / "rear.mp4"),
            (size, size),
            fps=fps,
            codec="libx264",
            quality=8,
            pix_fmt_out="yuv420p",
        )
        video.send(None)
        rear.send(None)
        try:
            for i in range(N):
                d.qpos[:3] = a["base_pos_w"][i]
                d.qpos[3:7] = a["base_quat_wxyz"][i]
                d.qpos[ids] = a["joint_pos"][i]
                mujoco.mj_forward(m, d)
                ims = []
                for key, label, az, el, dist in views:
                    cam.lookat[:] = p.world(p.edge - 0.05, 0, 0.9 if key == "rear" else 1.10)
                    cam.azimuth = az
                    cam.elevation = el
                    cam.distance = dist
                    renderer.update_scene(d, camera=cam)
                    im = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(im)
                    draw.rectangle((0, 0, size, round(36 * scale)), fill="#182029")
                    draw.text(
                        (round(10 * scale), round(6 * scale)),
                        f"{label}   f{i:03d}  {i / fps:.3f}s",
                        font=font,
                        fill="white",
                    )
                    draw.rectangle((0, round(610 * scale), size, size), fill="#182029")
                    draw.text(
                        (round(8 * scale), round(615 * scale)),
                        str(a["phase"][i]),
                        font=small,
                        fill="white",
                    )
                    ims.append(np.asarray(im))
                    pages[key].append(im.resize((320, 320)))
                    if (i + 1) % 24 == 0 or i == N - 1:
                        montage = Image.new("RGB", (1920, 1280), "#182029")
                        for j, tile in enumerate(pages[key]):
                            montage.paste(tile, ((j % 6) * 320, (j // 6) * 320))
                        montage.save(args.output / f"{key}_{i // 24:02d}.jpg", quality=90)
                        pages[key] = []
                video.send(np.concatenate(ims, axis=1))
                rear.send(ims[0])

                def local(v):
                    out = (v - p.box["center"]) @ p.R
                    out[..., 0] -= p.edge
                    out[..., 2] -= p.box["half_size"][2]
                    return out

                knees = np.array([d.xpos[m.body(s + "_knee_y_link").id] for s in ["l", "r"]])
                feet = np.array(
                    [
                        d.xpos[m.body(s + "_ankle_x_link").id]
                        + d.xmat[m.body(s + "_ankle_x_link").id].reshape(3, 3) @ p.sole
                        for s in ["l", "r"]
                    ]
                )
                hip = d.xpos[m.body("waist_z_link").id]
                footrot = [
                    p.R.T @ d.xmat[m.body(s + "_ankle_x_link").id].reshape(3, 3) for s in ["l", "r"]
                ]
                torso = p.R.T @ d.xmat[m.body("torso_link").id].reshape(3, 3)
                forearm = [
                    d.xpos[m.body(s + "_wrist_x_link").id]
                    + d.xmat[m.body(s + "_wrist_x_link").id].reshape(3, 3) @ p.armpts.mean(0)
                    for s in ["l", "r"]
                ]
                rows.append(
                    {
                        "frame": i,
                        "time_s": i / fps,
                        "phase": str(a["phase"][i]),
                        "feet_local_m": local(feet).tolist(),
                        "knees_local_m": local(knees).tolist(),
                        "pelvis_local_m": local(hip).tolist(),
                        "forearms_local_m": local(np.array(forearm)).tolist(),
                        "foot_heading_deg": [
                            float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) for R in footrot
                        ],
                        "foot_roll_pitch_yaw_deg": [
                            Rotation.from_matrix(R).as_euler("xyz", degrees=True).tolist()
                            for R in footrot
                        ],
                        "root_roll_pitch_yaw_deg": Rotation.from_matrix(torso)
                        .as_euler("xyz", degrees=True)
                        .tolist(),
                        "signed_left_minus_right_foot_y_m": float(
                            local(feet)[0, 1] - local(feet)[1, 1]
                        ),
                        "signed_left_minus_right_knee_y_m": float(
                            local(knees)[0, 1] - local(knees)[1, 1]
                        ),
                    }
                )
                if i % 60 == 0:
                    print(i, N, flush=True)
        finally:
            video.close()
            rear.close()
            renderer.close()
    report = {
        "motion": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "views": views,
        "camera": "Fixed in world coordinates for all frames; no camera following to conceal drift.",
        "frames": rows,
    }
    (args.output / "frame_metrics.json").write_text(json.dumps(report, indent=2))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
