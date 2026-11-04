#!/usr/bin/env python
"""Side-by-side videos: the same scenario under several controllers.

Each panel is one controller, rendered headless from the same fixed camera(s)
(Drake's VTK renderer, no Meshcat), labelled with the controller, time, phase,
the grasp being executed, and a live status: COLLISION / GRASPED / LIFTED /
SUCCESS.  Episodes keep running after a collision, so you can see what happens.

    python scripts/compare_videos.py --preset tall_blocker --out results/videos
    python scripts/compare_videos.py --scenarios results/stress_v3_heldout/scenarios.yaml \\
        --names mix_0012 mix_0034 --controllers naive st_rrt_nominal st_rrt --views front34 top

Writes <out>/<scenario>.mp4 (H.264) and <scenario>_final.png (last frame).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dyngrasp import scenario as S
from dyngrasp.baselines import HoldController, NaiveInterceptController
from dyngrasp.harness import advance, build_episode
from dyngrasp.planning.controller import SpaceTimeRRTController, STRRTConfig
from dyngrasp.scene import look_at_camera

VIEWS = {
    "front34": ([1.9, -1.5, 1.3], [0.35, -0.1, 0.35]),
    "close": ([1.35, -0.95, 0.85], [0.45, 0.0, 0.3]),  # the grasp zone, closer
    "side": ([0.3, -2.6, 0.9], [0.3, 0.0, 0.4]),
    "top": ([0.3, 0.0, 3.2], [0.3, 0.0, 0.0]),
    "robot_eye": ([-1.3, 0.0, 1.6], [0.45, 0.0, 0.3]),
}
CONTROLLERS = {
    "hold": ("hold (parked)", HoldController),
    "naive": ("naive: no avoidance", NaiveInterceptController),
    "st_rrt_nominal": ("(B) RRT, nominal grasp only", lambda: SpaceTimeRRTController(STRRTConfig(grasp_set="nominal"))),
    "st_rrt_noaccel": ("(B) before: no accel. limits", lambda: SpaceTimeRRTController(STRRTConfig(a_max=None))),
    "st_rrt": ("(B) now: accel.-limited, smooth", SpaceTimeRRTController),
}
PRESETS = {
    "tall_blocker": lambda: S.tall_blocker(0.1, 0.2),
    "plank_blocker": lambda: S.plank_blocker(0.1, 0.2),
    "tall_blocker_fast": lambda: S.tall_blocker(0.25, 0.4),
    "no_obstacles_fast": lambda: S.no_obstacles(0.3, "sugar_box"),
}
GREEN, RED, AMBER, WHITE = (60, 200, 90), (230, 60, 50), (240, 180, 40), (255, 255, 255)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _grasp_name(controller, h) -> str:
    plans = getattr(controller, "plans", None)
    active = h.controller_system._active
    if not plans or active is None or active.trajectory is None:
        return ""
    p = plans.get(id(active.trajectory))
    if p is None:
        return ""
    return p.goal.grasp_name if p.goal is not None else p.mode


def render_episode(spec, cname: str, views: list[str], fps: float, size, latency):
    label, factory = CONTROLLERS[cname]
    controller = factory()
    cams = {v: look_at_camera(v, *VIEWS[v], size[0], size[1]) for v in views}
    h = build_episode(spec, controller, cameras=cams, stop_on_collision=False, latency=latency)
    h.simulator.Initialize()
    station = h.scene.station
    f_big, f_small = _font(17), _font(13)
    frames = []
    acc_peak, obj_prev, obj_peak = 0.0, None, 0.0
    plant = h.scene.plant
    for t in np.arange(0.0, spec.duration + 1e-9, 1.0 / fps):
        advance(h, t)
        h.monitor.done = False  # keep going after a collision / success
        root = h.simulator.get_context()
        sc = station.GetMyContextFromRoot(root)
        tiles = [Image.fromarray(np.array(station.GetOutputPort(f"{v}.rgb_image").Eval(sc).data)[:, :, :3].copy())
                 for v in views]
        panel = Image.new("RGB", (size[0], size[1] * len(tiles) + 66), (25, 25, 30))
        for i, im in enumerate(tiles):
            panel.paste(im, (0, 66 + i * size[1]))
        d = ImageDraw.Draw(panel)
        m = h.monitor
        tt = root.get_time()
        phase = h.controller_system.log[-1].get("phase", "") if h.controller_system.log else ""
        grasp = _grasp_name(controller, h)
        d.text((8, 5), label, fill=WHITE, font=f_big)
        d.text((8, 30), f"t = {tt:4.1f} s   {phase}" + (f"   grasp: {grasp}" if grasp else ""), fill=(200, 200, 200), font=f_small)
        if m.collisions:
            status, col = f"COLLISION ({m.collisions[0].other_body}, t={m.collisions[0].t:.1f})", RED
        elif m.t_success is not None:
            status, col = f"SUCCESS (grasped t={m.t_grasp:.1f})", GREEN
        elif m.t_lift is not None:
            status, col = "LIFTED", GREEN
        elif m.t_grasp is not None:
            status, col = "GRASPED", AMBER
        else:
            status, col = "", WHITE
        if status:
            w = d.textlength(status, font=f_big)
            d.rectangle([size[0] - w - 18, 4, size[0] - 4, 28], fill=col)
            d.text((size[0] - w - 11, 5), status, fill=(0, 0, 0), font=f_big)
        clr = m.min_clearance_obstacles
        d.text((size[0] - 210, 34), f"min clearance {clr:.3f} m" if np.isfinite(clr) else "no obstacles",
               fill=(200, 200, 200), font=f_small)
        # live motion readout: commanded joint acceleration (now / peak) and object speed
        qc = h.controller_system.q_cmd_log
        dt = h.controller_system.controller.period
        acc = float(np.max(np.abs(qc[-1] - 2 * qc[-2] + qc[-3]))) / dt**2 if len(qc) >= 3 else 0.0
        if len(qc) >= 3:
            q = np.array(qc[-int(1.0 / fps / dt) - 2:])
            acc_frame = float(np.max(np.abs(np.diff(q, 2, axis=0)))) / dt**2 if len(q) >= 3 else acc
            acc_peak = max(acc_peak, acc_frame)
        p_obj = plant.EvalBodyPoseInWorld(plant.GetMyContextFromRoot(root), h.scene.target_body).translation()
        v_obj = 0.0 if obj_prev is None else float(np.linalg.norm(p_obj - obj_prev)) * fps
        obj_prev = p_obj.copy()
        obj_peak = max(obj_peak, v_obj)
        warn = RED if acc_peak > 20 else (200, 200, 200)
        d.text((8, 48), f"joint accel peak {acc_peak:6.1f} rad/s2   object speed {v_obj:4.2f} (peak {obj_peak:4.2f}) m/s",
               fill=warn, font=f_small)
        frames.append(panel)
    return frames


def write_video(spec, cnames, views, out: Path, fps=15, size=(480, 360), latency="measured"):
    per = [render_episode(spec, c, views, fps, size, latency) for c in cnames]
    n = min(len(f) for f in per)
    W = sum(f[0].size[0] for f in per) + 6 * (len(per) - 1)
    H = per[0][0].size[1] + 30
    title = (f"{spec.name}   object: {spec.target.object}   cart {spec.cart.speed:.2f} m/s   "
             f"{len(spec.obstacles)} obstacle(s)   latency: {latency}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{spec.name}.mp4"
    W2, H2 = W + (W % 2), H + (H % 2)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W2}x{H2}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )
    f_title = _font(15)
    last = None
    for i in range(n):
        canvas = Image.new("RGB", (W2, H2), (12, 12, 15))
        ImageDraw.Draw(canvas).text((8, 7), title, fill=WHITE, font=f_title)
        x = 0
        for f in per:
            canvas.paste(f[i], (x, 30))
            x += f[i].size[0] + 6
        ff.stdin.write(canvas.tobytes())
        last = canvas
    ff.stdin.close()
    ff.wait()
    last.save(out / f"{spec.name}_final.png")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", nargs="*", default=[], choices=sorted(PRESETS))
    ap.add_argument("--scenarios", help="YAML scenario set")
    ap.add_argument("--names", nargs="*", default=[], help="scenario names from --scenarios")
    ap.add_argument("--controllers", nargs="+", default=["naive", "st_rrt_nominal", "st_rrt"], choices=sorted(CONTROLLERS))
    ap.add_argument("--views", nargs="+", default=["front34"], choices=sorted(VIEWS))
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--latency", default="measured")
    ap.add_argument("--out", default="results/videos")
    args = ap.parse_args()
    latency = args.latency if args.latency in ("measured", "none") else float(args.latency)
    size = (args.width, int(args.width * 3 / 4))
    specs = [PRESETS[p]() for p in args.preset]
    if args.scenarios:
        by = {s.name: s for s in S.load_scenario_set(args.scenarios)}
        specs += [by[n] for n in args.names]
    for spec in specs:
        print("wrote", write_video(spec, args.controllers, args.views, Path(args.out), args.fps, size, latency), flush=True)


if __name__ == "__main__":
    main()
