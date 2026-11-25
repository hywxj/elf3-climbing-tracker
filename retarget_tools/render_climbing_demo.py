"""Render an existing kinematic ELF3 preview; never advances robot dynamics."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, tempfile
from pathlib import Path
import numpy as np, mujoco, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from view_motion import load, model_file

LABELS = {
    "READY": "站立准备",
    "GRIPPER_FLIP": "先翻转左手夹爪",
    "PREPARED": "夹爪调整完成",
    "WALK_LEFT_CLEAR": "接近高台：左脚迈步",
    "WALK_LEFT_LAND": "接近高台：左脚落地",
    "WALK_RIGHT_CLEAR": "接近高台：右脚跟进",
    "APPROACH_READY": "台前对齐",
    "ARM_HOVER": "抬臂越过台沿",
    "FOREARMS": "放下前臂与肘部",
    "FOREARM_HOLD": "双前臂与近肘部接触",
    "RAISE_AND_TUCK": "抬升骨盆，同时收右膝",
    "RIGHT_KNEE": "右膝移向台面",
    "LEFT_KNEE_CLEAR": "骨盆前移，左膝越过台沿",
    "BOTH_KNEES": "转换到双膝接触",
    "KNEEL_UP": "抬起上身，逐步释放前臂",
    "RIGHT_FOOT_CLEAR": "右脚收向台面",
    "RIGHT_FOOT": "右脚落台",
    "LEFT_FOOT_CLEAR": "左脚跟进",
    "CROUCH": "双脚落台",
    "STAND": "站起",
    "FINISHED": "完整演示结束",
}
LABELS.update(
    {
        "LEFT_FOOT_LIFT_OUTSIDE": "先在台外抬高整只左脚",
        "LEFT_FOOT_OVER_EDGE": "保持整脚高度，向前越过台沿",
        "LEFT_FOOT_LOWER_INSIDE": "脚跟进入台内后，再落脚",
    }
)
LABELS.update(
    {
        "PRELOAD": "双脚着地，屈腿下沉",
        "PUSH_OFF": "保持脚底接触，伸腿向前上方送髋",
        "ASSISTED_KNEE_TUCK": "前臂支撑下收膝（候选）",
        "KNEE_APPROACH": "膝部越沿并接近台面",
        "KNEE_ABSORB": "膝部接触过渡（候选）",
    }
)
LABELS.update(
    {
        "RIGHT_ARM_LOAD": "右前臂保留支撑，左臂移开",
        "RIGHT_FOOT_LIFT_OUTSIDE": "右脚先在台外抬高",
        "RIGHT_HAND_HOVER": "左膝与右脚支撑，调整右腕端",
        "RIGHT_HAND_PLANT": "右腕端建立接触",
        "RIGHT_FOOT_FLAT": "右脚全掌落稳",
        "SHIFT_RIGHT": "向右侧支撑移重心",
        "LEFT_FOOT_LAND": "左脚落台",
        "BOTH_FEET_LOAD": "双脚承重",
        "RELEASE_RIGHT_ARM": "撤右臂，协调起身",
    }
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    path, a = load(args.motion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fps = float(a["fps"])
    n = len(a["timestamps"])
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 25)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 19)
    snapshots = {
        0,
        "GRIPPER_FLIP",
        "FOREARM_HOLD",
        "RIGHT_KNEE",
        "BOTH_KNEES",
        "RIGHT_FOOT",
        "STAND",
        "FINISHED",
    }
    chosen = {0}
    for phase in snapshots - {0}:
        ix = np.flatnonzero(a["phase"] == phase)
        if len(ix):
            chosen.add(int(ix[-1]))
    if "takeoff_stage" in a:
        chosen = {0, len(a["timestamps"]) - 1}
        for stage in ["PRELOAD", "PUSH_OFF", "ASSISTED_KNEE_TUCK", "KNEE_ABSORB"]:
            ix = np.flatnonzero(a["takeoff_stage"] == stage)
            if len(ix):
                chosen.add(int(ix[-1]))
        for phase in ["FOREARM_HOLD", "BOTH_KNEES"]:
            ix = np.flatnonzero(a["phase"] == phase)
            if len(ix):
                chosen.add(int(ix[-1]))
    if "recovery_stage" in a:
        chosen = {0, n - 1}
        for phase in [
            "FOREARM_HOLD",
            "RIGHT_ARM_LOAD",
            "RIGHT_FOOT",
            "RIGHT_HAND_PLANT",
            "LEFT_FOOT_OVER_EDGE",
            "RELEASE_RIGHT_ARM",
        ]:
            ix = np.flatnonzero(a["phase"] == phase)
            if len(ix):
                chosen.add(int(ix[-1]))
    if str(a.get("support_mode", "")) == "v4_forearms_to_knees_to_feet":
        chosen = {0, n - 1}
        for phase in [
            "FOREARM_HOLD",
            "RIGHT_KNEE",
            "BOTH_KNEES",
            "KNEEL_UP",
            "RIGHT_FOOT",
            "CROUCH",
        ]:
            ix = np.flatnonzero(a["phase"] == phase)
            if len(ix):
                chosen.add(int(ix[-1]))
    stills = []
    with tempfile.TemporaryDirectory(prefix="elf3_demo_render_") as folder:
        m = mujoco.MjModel.from_xml_path(str(model_file(folder, a, True)))
        m.vis.global_.offwidth = 960
        m.vis.global_.offheight = 720
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.65
        d = mujoco.MjData(m)
        ids = [m.jnt_qposadr[m.joint(str(name)).id] for name in a["joint_names"]]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = a.get("preview_camera_lookat_w", np.median(a["base_pos_w"], axis=0))
        cam.distance = float(a.get("preview_camera_distance", 3.15))
        cam.azimuth = 130
        cam.elevation = -15
        renderer = mujoco.Renderer(m, height=720, width=960)
        writer = imageio_ffmpeg.write_frames(
            str(args.output),
            (960, 720),
            fps=fps,
            codec="libx264",
            pix_fmt_out="yuv420p",
            quality=8,
            macro_block_size=16,
        )
        writer.send(None)
        try:
            for i in range(n):
                d.qpos[:3] = a["base_pos_w"][i]
                d.qpos[3:7] = a["base_quat_wxyz"][i]
                d.qpos[ids] = a["joint_pos"][i]
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera=cam)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 960, 58), fill="#182029")
                phase = str(a["phase"][i])
                phase = (
                    str(a["takeoff_stage"][i])
                    if "takeoff_stage" in a and str(a["takeoff_stage"][i])
                    else phase
                )
                phase = (
                    str(a["foot_transfer_stage"][i])
                    if "foot_transfer_stage" in a and str(a["foot_transfer_stage"][i])
                    else phase
                )
                phase = (
                    str(a["recovery_stage"][i])
                    if "recovery_stage" in a and str(a["recovery_stage"][i])
                    else phase
                )
                draw.text(
                    (20, 12),
                    f"{i / fps:05.2f}s  /  {(n - 1) / fps:.2f}s     " + LABELS.get(phase, phase),
                    font=font,
                    fill="white",
                )
                draw.rectangle((0, 678, 960, 720), fill="#182029")
                draw.text(
                    (20, 688),
                    str(a.get("preview_warning", "运动学动作演示 · 未验证动力学"))
                    + "     橙色：左手夹爪包络",
                    font=small,
                    fill="#dce5ef",
                )
                writer.send(np.asarray(im))
                if i in chosen:
                    stills.append(im.resize((480, 360)))
                if i % 150 == 0:
                    print(f"Rendered {i}/{n}", flush=True)
        finally:
            writer.close()
            renderer.close()
    montage = Image.new("RGB", (4 * 480, 2 * 360), "#182029")
    for i, im in enumerate(stills[:8]):
        montage.paste(im, ((i % 4) * 480, (i // 4) * 360))
    montage.save(args.output.with_suffix(".png"))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
