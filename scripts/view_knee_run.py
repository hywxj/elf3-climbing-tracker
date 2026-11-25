"""Play one saved knee-entry policy with Isaac Sim and write its measured contact trace."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from elf3_climbing.paths import OUTPUT


def training_pid(run):
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmdline.read_bytes().split(b"\0")
            if b"elf3_climbing.cli" not in args or b"--mode" not in args or b"--output" not in args:
                continue
            if args[args.index(b"--mode") + 1] != b"train":
                continue
            target = Path(os.fsdecode(args[args.index(b"--output") + 1])).expanduser()
            if not target.is_absolute():
                target = (cmdline.parent / "cwd").resolve() / target
            if target.resolve() == run:
                return cmdline.parent.name
        except (OSError, IndexError):
            continue
    return None


def checkpoint_path(run, selection):
    if selection in ("latest", "best"):
        return run / f"{selection}.pt"
    iteration = int(selection)
    numbered = run / f"checkpoint_{iteration:05d}.pt"
    if numbered.is_file():
        return numbered
    # Old runs saved zero-based PPO indices. Evaluation is written after model serialization.
    legacy = run / "ppo" / f"model_{iteration - 1}.pt"
    if legacy.is_file() and any(
        int(path.stem.split("_")[-1]) >= iteration for path in run.glob("evaluation_[0-9]*.json")
    ):
        return legacy
    return numbered


def wait_for_training(run, selection):
    numbered = selection.isdecimal()
    if not (run / "completion.json").exists():
        target = f"第 {selection} 轮检查点" if numbered else "训练结束"
        print(f"等待{target}：{run.name}，等待时不占用 GPU。", flush=True)
    while not (run / "completion.json").exists():
        if numbered and checkpoint_path(run, selection).is_file():
            return
        if (run / "error.json").exists():
            raise SystemExit(
                f"训练报错。查看 {run / 'error.json'}，去掉 --wait 可回放已保存的检查点。"
            )
        if training_pid(run) is None and not (run / "completion.json").exists():
            raise SystemExit("训练进程已退出且没有完成记录。去掉 --wait 可回放已保存的检查点。")
        time.sleep(5)
    completion = json.loads((run / "completion.json").read_text())
    if completion.get("status") != "COMPLETED":
        raise SystemExit(f"训练退出状态：{completion.get('status')}")
    result = run / "training_result.json"
    if result.exists():
        saved = json.loads(result.read_text())
        print(
            f"训练已结束：{saved.get('ppo_iterations')} 轮，{saved.get('stop_reason')}。",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", help="Run name under runtime_v1/runs or its absolute path")
    p.add_argument("--checkpoint", default="best",
                   help="best (default, by the training run's scoring rules), latest, or a saved iteration")
    p.add_argument("--phase", type=float, default=0.0)
    p.add_argument("--speed", type=float, default=0.25)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--view", choices=("right", "left", "front", "rear-left"), default="right")
    p.add_argument(
        "--wait", action="store_true", help="Wait for the requested iteration, or for the run to finish when using latest/best"
    )
    p.add_argument("--headless", action="store_true", help="Write the same traces without a window")
    args = p.parse_args()
    run = (OUTPUT / Path(args.run).expanduser()).resolve()
    request = run / "request.json"
    if not request.is_file():
        p.error(f"Training request missing: {request}")
    req = json.loads(request.read_text())
    if req.get("segment") != "knee_entry" or req.get("mode") != "train":
        p.error("Select a knee-entry training run")
    if not 0 <= args.phase <= 1 or not 0 < args.speed < float("inf") or args.episodes < 0:
        p.error("Expected phase in [0,1], finite positive speed and nonnegative episodes")
    if args.checkpoint not in ("latest", "best"):
        if not args.checkpoint.isascii() or not args.checkpoint.isdecimal() or int(args.checkpoint) < 1:
            p.error("Checkpoint must be latest, best, or a positive iteration")
        if int(args.checkpoint) > req["iterations"]:
            p.error(f"Requested iteration exceeds the run budget of {req['iterations']}")
        contract = run / "training_contract.json"
        if contract.is_file():
            interval = json.loads(contract.read_text()).get("checkpoint_interval")
            if interval and int(args.checkpoint) % interval:
                p.error(f"This run archives every {interval} iterations. Use latest for its most recent evaluation.")
    if args.wait:
        wait_for_training(run, args.checkpoint)
    checkpoint = checkpoint_path(run, args.checkpoint)
    if not checkpoint.is_file():
        p.error(f"Checkpoint missing: {checkpoint}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = OUTPUT / f"play_{run.name}_{args.checkpoint}_p{args.phase:g}_{stamp}"
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        "--mode",
        "play",
        "--segment",
        "knee_entry",
        "--reference",
        req["reference"],
        "--knee-support-rule",
        req.get("knee_support_rule", "continuous"),
        "--knee-wrist-contact",
        req.get("knee_wrist_contact", "distal-only"),
        "--checkpoint",
        str(checkpoint),
        "--num-envs",
        "1",
        "--device",
        req["device"],
        "--seed",
        str(req["seed"]),
        "--play-start-phase",
        str(args.phase),
        "--play-speed",
        str(args.speed),
        "--play-episodes",
        str(args.episodes),
        "--view",
        args.view,
        "--output",
        str(output),
    ]
    if req.get("preload_example"):
        cmd.extend(["--preload-example", req["preload_example"]])
    if req.get("knee_transfer_guide"):
        cmd.extend(["--knee-transfer-guide", req["knee_transfer_guide"]])
    config = run / "input_config.toml"
    if config.is_file():
        cmd.extend(["--config", str(config)])
    if args.headless:
        cmd.append("--headless")
    print(f"检查点：{checkpoint}\n接触记录：{output / 'knee_contacts.csv'}", flush=True)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "third_party/rsl_rl"), os.environ.get("PYTHONPATH", "")]
    )
    os.chdir(ROOT)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
