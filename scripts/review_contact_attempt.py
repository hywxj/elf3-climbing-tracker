"""Render an Isaac rollout's logged poses at 0.10 s intervals, using the matching robot mesh."""

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from elf3_climbing.paths import OUTPUT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--video", type=Path, help="Also export the recorded rollout at real speed")
    p.add_argument(
        "--episode", type=int, help="Select a physical episode from an offline example pack"
    )
    p.add_argument(
        "--model", type=Path, default=OUTPUT / "mount_070_ground_to_steady_rise_20261123/model.xml"
    )
    args = p.parse_args()
    with np.load(args.trace, allow_pickle=False) as data:
        poses, joints = data["root_pose"], data["joint_pos"]
        names = data["joint_names"]
        if args.episode is None:
            metrics = dict(zip(data["metric_names"], data["metrics"].T, strict=True))
            dt, lane = float(data["dt"]), int(data["lane"])
            ts = (np.arange(len(poses)) + 1) * dt
            load = metrics["right_knee_top_N"]
            local = (
                "start" in data and data["start"].item() == "archived_knee_start"
            ) or args.trace.stem.startswith("knee_seed_")
            title = "统一策略 · 扶台起步局部复测" if local else "统一策略 · Isaac 地面复测"
        else:
            mask = data["episode_id"] == args.episode
            if not mask.any():
                p.error("Episode missing from example pack")
            dt, lane = float(data["step_dt"]), args.episode
            poses, joints = poses[mask], joints[mask]
            ts = data["step"][mask] * dt
            load = data["student_contact_force"][mask, 7]
            title = "离线教师示范 · Isaac 实际姿态"
    document = ET.parse(args.model)
    document.find("compiler").set("meshdir", str(args.model.parent / "meshes"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(document.getroot(), encoding="unicode"))
    addr = [model.jnt_qposadr[model.joint(str(name)).id] for name in names]
    state = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-1.0, -0.08, 0.66]
    camera.distance, camera.azimuth, camera.elevation = 2.95, 79, -11
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    option.sitegroup[:] = 0
    model.vis.headlight.ambient[:] = 0.63
    model.geom_rgba[model.geom("preview_box_0").id] = [0.30, 0.46, 0.51, 1]
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17, index=2)
    stride = round(0.10 / dt)
    selected = sorted(set([0, *np.flatnonzero(np.round(ts / dt).astype(int) % stride == 0)]))
    if selected[-1] != len(poses) - 1:
        selected.append(len(poses) - 1)
    w, h, cols = 234, 145, 5
    if args.video:
        import imageio_ffmpeg

        vw, vh = 768, 448
        model.vis.global_.offwidth, model.vis.global_.offheight = vw, vh
        writer = imageio_ffmpeg.write_frames(
            str(args.video),
            (vw, vh + 48),
            fps=1 / dt,
            codec="libx264",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "20", "-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            with mujoco.Renderer(model, height=vh, width=vw) as renderer:
                for i, t in enumerate(ts):
                    state.qpos[:7], state.qpos[addr] = poses[i], joints[i]
                    mujoco.mj_forward(model, state)
                    renderer.update_scene(state, camera=camera, scene_option=option)
                    frame = Image.new("RGB", (vw, vh + 48), "#f7f8f7")
                    frame.paste(Image.fromarray(renderer.render()), (0, 48))
                    ImageDraw.Draw(frame).text(
                        (12, 12),
                        f"{title}  |  {t:.2f} s  |  右膝台面载荷 {load[i]:.0f} N",
                        fill="#203e40",
                        font=font,
                    )
                    writer.send(np.asarray(frame))
        finally:
            writer.close()
    with mujoco.Renderer(model, height=h, width=w) as renderer:
        for part in range(0, len(selected), 25):
            frames = selected[part : part + 25]
            sheet = Image.new(
                "RGB", (w * cols, 40 + (h + 30) * ((len(frames) + cols - 1) // cols)), "#f7f8f7"
            )
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (8, 4),
                f"{title}  |  环境 {lane}  |  右膝台面载荷（N）",
                fill="#203e40",
                font=font,
            )
            for k, i in enumerate(frames):
                state.qpos[:7], state.qpos[addr] = poses[i], joints[i]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera=camera, scene_option=option)
                x, y = (k % cols) * w, 40 + (k // cols) * (h + 30)
                sheet.paste(Image.fromarray(renderer.render()), (x, y))
                draw.text(
                    (x + 6, y + h),
                    f"{ts[i]:.2f} s     {load[i]:.0f} N",
                    font=font,
                    fill="#1f4045",
                )
            output = args.trace.with_name(f"{args.trace.stem}_frames_{part // 25 + 1}.jpg")
            sheet.save(output, quality=91)
            print(output)


if __name__ == "__main__":
    main()
