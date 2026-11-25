"""Recheck foot orientation in a recorded MuJoCo mount and enlarge the original video."""

import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def longest_seconds(mask, dt):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return float(np.diff(edges.reshape(-1, 2), axis=1).max(initial=0) * dt)


def measure(directory):
    data = np.load(directory / "pose_trace.npz")
    dt = float(data["dt"])
    metrics = data["metrics"]
    model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
    state = mujoco.MjData(model)
    sites = [model.site(name).id for name in ("left_foot", "right_foot")]
    tilt = np.empty((len(metrics), 2))
    spacing = np.empty(len(metrics))
    for i, qpos in enumerate(data["qpos"][:-1]):
        state.qpos[:] = qpos
        mujoco.mj_kinematics(model, state)
        # The foot sites are aligned with the ankle_x links. Local +Z is the sole normal.
        tilt[i] = np.degrees(np.arccos(state.site_xmat[sites, 8].clip(-1, 1)))
        spacing[i] = np.linalg.norm(np.diff(state.site_xpos[sites, :2], axis=0))
    accepted = metrics[:, 1].astype(bool)
    report = {
        "pose_trace": str(directory / "pose_trace.npz"),
        "dt_s": dt,
        "standing_original_longest_s": longest_seconds(accepted, dt),
        "standing_bilateral_sole_tilt_max_15deg_longest_s": longest_seconds(
            accepted & (tilt.max(axis=1) <= 15), dt
        ),
        "standing_bilateral_sole_tilt_max_20deg_longest_s": longest_seconds(
            accepted & (tilt.max(axis=1) <= 20), dt
        ),
        "median_sole_tilt_deg_lr_during_original_standing": np.median(
            tilt[accepted], axis=0
        ).tolist(),
        "observations": [],
    }
    for t in (0, 0.685, 2.0, 3.2, 4.855, metrics[-1, 0]):
        i = round(t / dt)
        row = metrics[i]
        report["observations"].append(
            dict(
                time_s=t,
                hip_above_top_m=row[3],
                foot_top_N_lr=row[5:7].tolist(),
                floor_N=row[7],
                other_contact_N=row[8],
                foot_tilt_deg_lr=tilt[i].tolist(),
                horizontal_foot_spacing_m=spacing[i],
            )
        )
    return dt, metrics, tilt, spacing, report


def render(video, output, measurements):
    dt, metrics, angles, spacing, _ = measurements
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    meta = next(reader)
    if tuple(meta["size"]) != (1280, 736) or not math.isclose(meta["fps"], 25):
        raise ValueError("Expected the 1280×736 25 fps companion video of this MuJoCo rollout")
    fonts = {}
    for bold in (True, False):
        weight = "Bold" if bold else "Regular"
        path = f"/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc"
        for px in (16, 18, 20, 22, 24, 26, 30, 33):
            fonts[px, bold] = ImageFont.truetype(path, px, index=2)
    out_tmp = output.with_name(output.stem + ".writing.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(out_tmp),
        (1280, 736),
        fps=25,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"],
    )
    writer.send(None)
    ink = "#182f38"
    white = "#fbfcfc"
    teal = "#176956"
    muted = "#546b74"
    try:
        for k, frame in enumerate(reader):
            if k > 130:
                break
            t = k / 25
            i = round(t / dt)
            row = metrics[i]
            canvas = Image.new("RGB", (1280, 736), white)
            draw = ImageDraw.Draw(canvas)

            def label(x, y, txt, size=20, bold=False, color=ink, anchor="lt"):
                draw.text((x, y), txt, font=fonts[size, bold], fill=color, anchor=anchor)

            label(22, 20, "前段复核 · 送膝之后的 5.2 秒", 33, True)
            label(
                24, 67, "原 MuJoCo 回放局部放大　｜　起点为送膝姿态，初始速度为零", 18, False, muted
            )
            label(1254, 24, f"仿真时间  {t:.2f} s", 26, True, anchor="rt")
            label(1254, 68, "0.5 倍慢放", 18, False, teal, anchor="rt")
            full = Image.frombytes("RGB", (1280, 736), frame)
            # Exclude the old score and captions. The original camera follows the robot.
            view = full.crop((114, 100, 861, 560)).resize((877, 540), Image.Resampling.LANCZOS)
            canvas.paste(view, (20, 105))
            x, right = 920, 1254
            label(x, 116, "台面脚部支撑力", 24, True)
            for j, side in enumerate(("左脚", "右脚")):
                yy = 162 + 45 * j
                label(x, yy, side, 20, False, muted)
                label(right, yy, f"{row[5 + j]:.0f} N", 24, True, anchor="rt")
            draw.line((x, 255, right, 255), fill="#c7d2d5", width=2)
            label(x, 281, "其余接触垂向载荷", 20, False, muted)
            label(right, 279, f"{row[8]:.0f} N", 24, True, anchor="rt")
            label(x, 335, "髋部高出台面", 20, False, muted)
            label(right, 333, f"{row[3]:.2f} m", 24, True, anchor="rt")
            draw.line((x, 384, right, 384), fill="#c7d2d5", width=2)
            label(x, 410, "足底法线偏离竖直", 22, True)
            label(x, 461, "左 / 右", 20, False, muted)
            label(right, 458, f"{angles[i, 0]:.0f}° / {angles[i, 1]:.0f}°", 26, True, anchor="rt")
            label(x, 529, "两脚测点水平距离", 20, False, muted)
            label(right, 527, f"{spacing[i]:.2f} m", 24, True, anchor="rt")
            if t < 0.48:
                note = "起点已在台沿。此时双脚对台面的支撑力均为零。"
            elif t < 1.36:
                note = "0.68 s，右腕与台面碰撞模型重叠约 2.5 cm。"
            elif t < 2.64:
                note = "2.00 s，躯干已抬起，双脚仍为 0 N，大腿附近与右腕合计承重约 399 N。"
            elif t < 3.8:
                note = "3.20 s，双脚有载荷，左脚底倾斜 96°，大腿和右腕仍在支撑。"
            else:
                note = "4.855 s 首次进入原记录的连续站稳段。此刻左、右脚底仍倾斜 15° 和 19°。"
            draw.line((20, 655, 1258, 655), fill="#c7d2d5", width=3)
            draw.line((20, 655, 20 + int(1238 * t / 5.2), 655), fill=teal, width=3)
            label(22, 679, note, 22, True)
            payload = canvas.tobytes()
            writer.send(payload)
            if k < 130:
                writer.send(payload)
    finally:
        reader.close()
        writer.close()
    out_tmp.replace(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rollout", type=Path)
    ap.add_argument("--audit-json", type=Path)
    ap.add_argument("--original-video", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = measure(args.rollout)
    if args.audit_json:
        args.audit_json.write_text(json.dumps(result[-1], indent=2, ensure_ascii=False) + "\n")
    if args.original_video or args.output:
        if not (args.original_video and args.output):
            ap.error("--original-video and --output are required together")
        render(args.original_video, args.output, result)
