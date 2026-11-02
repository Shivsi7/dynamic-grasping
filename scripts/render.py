#!/usr/bin/env python
"""Headless renders of scenarios: snapshots from fixed cameras and an animated GIF.

    python scripts/render.py --out results/renders        # all presets, contact sheets + gif
    python scripts/render.py --scenario scenarios/dev_v0.yaml --index 3 --times 0 3 4 5

No Meshcat, no browser: the images come from Drake's VTK renderer through
cameras added to the station.  Each snapshot is annotated with the time, the
controller phase, and the scripted vs. simulated position of every moving body.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from dyngrasp import scenario as S  # noqa: E402
from dyngrasp.baselines import HoldController, NaiveInterceptController  # noqa: E402
from dyngrasp.harness import advance, build_episode  # noqa: E402
from dyngrasp.scene import look_at_camera  # noqa: E402

# name -> (camera position, look-at point)
VIEWS = {
    "front34": ([1.9, -1.5, 1.3], [0.35, -0.1, 0.35]),  # from the +x,-y side, looking back at the robot
    "side": ([0.3, -2.6, 0.9], [0.3, 0.0, 0.4]),  # along +y: the corridor and the lane edge-on
    "top": ([0.3, 0.0, 3.2], [0.3, 0.0, 0.0]),  # straight down
    "robot_eye": ([-1.3, 0.0, 1.6], [0.45, 0.0, 0.3]),  # from behind the robot towards the cart
}
CONTROLLERS = {"hold": HoldController, "naive": NaiveInterceptController}


def snapshots(spec, controller, times, views, size=(640, 480)):
    cams = {name: look_at_camera(name, p, c, size[0], size[1]) for name, (p, c) in views.items()}
    h = build_episode(spec, controller, cameras=cams, stop_on_collision=False)
    h.simulator.Initialize()
    station = h.scene.station
    plant = h.scene.plant
    frames = []
    for t in times:
        advance(h, t)
        h.monitor.done = False  # keep rendering past a success or a collision
        root = h.simulator.get_context()
        sc = station.GetMyContextFromRoot(root)
        pc = plant.GetMyContextFromRoot(root)
        tt = root.get_time()
        # scripted vs simulated
        lines = [f"t = {tt:.2f} s   controller: {controller.name}"]
        if h.controller_system.log:
            lines[0] += f"   phase: {h.controller_system.log[-1].get('phase', '?')}"
        c_sim = plant.EvalBodyPoseInWorld(pc, h.scene.cart_top_body).translation()
        c_scr = h.world.cart_pose(tt).translation()
        o_sim = plant.EvalBodyPoseInWorld(pc, h.scene.target_body).translation()
        o_scr = h.world.target_body_pose(tt).translation()
        lines.append(f"cart   sim {np.round(c_sim, 3)}  script {np.round(c_scr, 3)}")
        lines.append(f"object sim {np.round(o_sim, 3)}  script {np.round(o_scr, 3)}")
        for o, p in zip(spec.obstacles, h.world.obstacle_positions(tt)):
            b = plant.EvalBodyPoseInWorld(pc, h.scene.obstacle_body(o.name)).translation()
            lines.append(f"{o.name:10s} sim {np.round(b, 3)}  script {np.round(p, 3)}")
        m = h.monitor
        lines.append(
            f"collisions: {len(m.collisions)}   min clearance to obstacles: {m.min_clearance_obstacles:.3f} m"
            f"   grasped: {m.t_grasp is not None}   lifted: {m.t_lift is not None}"
        )
        imgs = {}
        for name in cams:
            a = np.array(station.GetOutputPort(f"{name}.rgb_image").Eval(sc).data)[:, :, :3]
            imgs[name] = Image.fromarray(a.copy())
        frames.append((tt, imgs, lines))
    return frames


def contact_sheet(frames, views, path, font=None):
    w, hh = next(iter(frames[0][1].values())).size
    text_h = 15 * max(len(f[2]) for f in frames) + 10
    sheet = Image.new("RGB", (w * len(views), (hh + text_h) * len(frames)), "white")
    for r, (t, imgs, lines) in enumerate(frames):
        y0 = r * (hh + text_h)
        for c, name in enumerate(views):
            sheet.paste(imgs[name], (c * w, y0))
            ImageDraw.Draw(sheet).text((c * w + 6, y0 + 4), name, fill="white")
        d = ImageDraw.Draw(sheet)
        for i, line in enumerate(lines):
            d.text((8, y0 + hh + 4 + 15 * i), line, fill="black")
    sheet.save(path)
    return path


def gif(spec, controller, path, view="front34", fps=10, size=(480, 360)):
    times = np.arange(0.0, spec.duration + 1e-9, 1.0 / fps)
    frames = snapshots(spec, controller, times, {view: VIEWS[view]}, size)
    imgs = []
    for t, im, lines in frames:
        im = im[view].copy()
        ImageDraw.Draw(im).text((6, 4), lines[0], fill="white")
        ImageDraw.Draw(im).text((6, size[1] - 16), lines[-1][:70], fill="white")
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/renders")
    ap.add_argument("--scenario", help="YAML scenario set")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--controller", default="naive", choices=sorted(CONTROLLERS))
    ap.add_argument("--times", nargs="+", type=float, default=None)
    ap.add_argument("--gif", action="store_true", help="also write an animated gif of the episode")
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    views = {k: VIEWS[k] for k in args.views}

    if args.scenario:
        jobs = [(S.load_scenario_set(args.scenario)[args.index], args.controller)]
    else:
        jobs = [
            (S.no_obstacles(0.15, object="sugar_box"), "naive"),
            (S.tall_blocker(0.1, 0.2), "naive"),
            (S.plank_blocker(0.1, 0.2), "hold"),
            (S.sample_scenario(np.random.default_rng(5), n_obstacles=4, name="random_s5", seed=5), "hold"),
        ]
    for spec, cname in jobs:
        times = args.times or sorted({0.0, spec.cart.t_enter, spec.cart.t_enter + 1.0, spec.cart.t_enter + 2.0, spec.cart.t_closest, min(spec.duration, spec.cart.t_exit)})
        frames = snapshots(spec, CONTROLLERS[cname](), times, views)
        p = contact_sheet(frames, views, out / f"{spec.name}__{cname}.png")
        print("wrote", p)
        if args.gif:
            print("wrote", gif(spec, CONTROLLERS[cname](), out / f"{spec.name}__{cname}.gif"))


if __name__ == "__main__":
    main()
