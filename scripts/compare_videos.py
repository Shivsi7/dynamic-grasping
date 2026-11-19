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

    # a (D) policy next to (C), and the report's frame strip at t = 6 s
    python scripts/compare_videos.py --scenarios scenarios/preset_grids.yaml --names NAME \\
        --controllers rl:runs/ppo/best hybrid --views front34 top --frames 6.0 --frames-out docs/report/frames

`rl:<dir>` is a (D) checkpoint (RLController).  --frames T also writes <frames-out>/<scenario>_0.jpg (the scene at
t = 0) and <scenario>_<k>.jpg (panel k's first view at t = T).

    # one video per scene that (B) did not grasp, next to the certification evidence
    python scripts/compare_videos.py --certify results/certify_v12/certify.csv \
        --scenarios results/stress_v3_heldout/scenarios.yaml --jobs 5 --out results/videos_certify

`oracle_<s>` panels are the certification oracle (scripts/certify.py): the same planner
with 20x the compute and planning time free (latency "none"), at joint-acceleration limits
<s> x IIWA_ACCEL_REF (per joint).
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
from dyngrasp.harness import advance, build_episode, summarize
from dyngrasp.hybrid import HybridController
from dyngrasp.planning.controller import SpaceTimeRRTController
from dyngrasp.planning.variants import VARIANTS, Variant
from dyngrasp.reactive import ReactiveQPController
from dyngrasp.rl.controller import RLController
from dyngrasp.scene import look_at_camera

VIEWS = {
    "front34": ([1.9, -1.5, 1.3], [0.35, -0.1, 0.35]),
    "close": ([1.35, -0.95, 0.85], [0.45, 0.0, 0.3]),  # the grasp zone, closer
    "side": ([0.3, -2.6, 0.9], [0.3, 0.0, 0.4]),
    "top": ([0.3, 0.0, 3.2], [0.3, 0.0, 0.0]),
    "robot_eye": ([-1.3, 0.0, 1.6], [0.45, 0.0, 0.3]),
}


def _oracle(scale: float, accel_limits: bool = True):
    from certify import oracle_config  # scripts/certify.py, next to this script

    return lambda: SpaceTimeRRTController(oracle_config(accel_limits, 20.0, scale))


# name -> Variant(label, factory, episode_kw): the (B) variants, the baselines, and the
# certification oracle (planning "offline": its planning time is free, latency "none")
CONTROLLERS = {
    "hold": Variant("hold (parked)", HoldController),
    "naive": Variant("naive: no avoidance", NaiveInterceptController),
    "reactive": Variant("(A) reactive QP", ReactiveQPController),
    "hybrid": Variant("(C) hybrid trajopt", HybridController),
    **VARIANTS,
    **{f"oracle_{s}": Variant(f"oracle @ {s} x ref. accel. (20x compute)", _oracle(s), {"latency": "none"})
       for s in (0.6, 0.8, 1.0, 1.5)},
    "oracle_noaccel": Variant("oracle, NO accel. limits (20x compute)", _oracle(0.6, False), {"latency": "none"}),
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


def _variant(cname: str) -> Variant:
    if cname.startswith("rl:"):
        return Variant("(D) PPO policy", lambda: RLController(cname[3:]))
    return CONTROLLERS[cname]


def render_episode(spec, cname: str, views: list[str], fps: float, size, latency):
    variant = _variant(cname)
    label, controller = variant.label, variant.factory()
    cams = {v: look_at_camera(v, *VIEWS[v], size[0], size[1]) for v in views}
    h = build_episode(spec, controller, cameras=cams, stop_on_collision=False,
                      **{"latency": latency, **variant.episode_kw})
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
    r = summarize(spec, controller, h, h.simulator.get_context().get_time(), 0.0, False)
    controller.close()
    print(f"{spec.name} {label}: {r.outcome} (grasp {r.t_grasp}, collision {r.t_first_collision} {r.collision_with})",
          flush=True)
    return frames


def write_frames(spec, per, t: float, fps: float, size, out: Path) -> None:
    """<scenario>_0.jpg: the scene at t = 0 (panel 0's first view); <scenario>_<k>.jpg: panel k's at time t."""
    out.mkdir(parents=True, exist_ok=True)
    box = (0, 0, size[0], 66 + size[1])
    first = per[0][0].crop(box)
    d = ImageDraw.Draw(first)
    d.rectangle([0, 0, size[0], 65], fill=(25, 25, 30))
    d.text((8, 5), "scene, t = 0.0 s", fill=WHITE, font=_font(17))
    d.text((8, 30), spec.name, fill=(200, 200, 200), font=_font(13))
    d.text((8, 48), f"object: {spec.target.object}   cart {spec.cart.speed:.2f} m/s   {len(spec.obstacles)} obstacle(s)",
           fill=(200, 200, 200), font=_font(13))
    first.save(out / f"{spec.name}_0.jpg")
    for k, f in enumerate(per, 1):
        f[min(round(t * fps), len(f) - 1)].crop(box).save(out / f"{spec.name}_{k}.jpg")


def write_video(spec, cnames, views, out: Path, fps=15, size=(480, 360), latency="measured", note="",
                frames_t=None, frames_out=None):
    per = [render_episode(spec, c, views, fps, size, latency) for c in cnames]
    if frames_t is not None:
        write_frames(spec, per, frames_t, fps, size, Path(frames_out or out))
    n = min(len(f) for f in per)
    W = sum(f[0].size[0] for f in per) + 6 * (len(per) - 1)
    H = per[0][0].size[1] + 30
    title = (f"{spec.name}   object: {spec.target.object}   cart {spec.cart.speed:.2f} m/s   "
             f"{len(spec.obstacles)} obstacle(s)" + (f"   |   {note}" if note else f"   latency: {latency}"))
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
    ap.add_argument("--controllers", nargs="+", default=["naive", "st_rrt_nominal", "st_rrt"],
                    help=f"{', '.join(sorted(CONTROLLERS))} or rl:<checkpoint dir>")
    ap.add_argument("--views", nargs="+", default=["front34"], choices=sorted(VIEWS))
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--latency", default="measured")
    ap.add_argument("--out", default="results/videos")
    ap.add_argument("--certify", help="certify.csv: one video per scene, (B) now vs. the certification evidence")
    ap.add_argument("--jobs", type=int, default=1, help="render scenes in parallel processes")
    ap.add_argument("--frames", type=float, help="also write the report's frames at this time (s)")
    ap.add_argument("--frames-out", help="where (default --out)")
    args = ap.parse_args()
    latency = args.latency if args.latency in ("measured", "none") else float(args.latency)
    size = (args.width, int(args.width * 3 / 4))
    specs = [PRESETS[p]() for p in args.preset]
    if args.scenarios:
        by = {s.name: s for s in S.load_scenario_set(args.scenarios)}
        specs += [by[n] for n in args.names]
    jobs = [(spec, args.controllers, "") for spec in specs]
    if args.certify:
        import pandas as pd

        by = {s.name: s for s in S.load_scenario_set(args.scenarios)}
        for r in pd.read_csv(args.certify).sort_values("scenario").itertuples():
            a = r.planner_accel_needed
            if np.isfinite(a):
                evidence, verdict = f"oracle_{a:g}", f"possible at {a:g} x ref. accel."
                if r.oracle_success:
                    verdict = "possible at the current limits: online planner MISSED it"
            else:
                evidence, verdict = "oracle_noaccel", "impossible at any acceleration (" + (
                    "no feasible grasp)" if r.n_goals_without_tail_accel_limit == 0 else "no plan even unconstrained)")
            jobs.append((by[r.scenario], ["st_rrt", evidence], f"verdict: {verdict}"))
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(args.jobs) as ex:
            futs = [ex.submit(write_video, spec, cn, args.views, Path(args.out), args.fps, size, latency, note,
                              args.frames, args.frames_out) for spec, cn, note in jobs]
            for f in as_completed(futs):
                print("wrote", f.result(), flush=True)
    else:
        for spec, cn, note in jobs:
            print("wrote", write_video(spec, cn, args.views, Path(args.out), args.fps, size, latency, note,
                                       args.frames, args.frames_out), flush=True)


if __name__ == "__main__":
    main()
