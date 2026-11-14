"""(D) reward terms; their weights are config["reward"].

progress   k_p (gamma Phi_p(s') - Phi_p(s)),  Phi_p = -(|p_grasp - p_G| + rot_weight x rotation error in rad).
           The grasp pose is the WorldModel's (as in the observation).  While the object is in hand (the
           monitor's test) Phi_p = 0, so carrying it away from the cart-borne grasp pose (lifting) is not
           penalised; this keeps the term a potential difference (of the state including "in hand").
clearance  gamma Phi_c(s') - Phi_c(s),  Phi_c = -weight x max(0, d_clear - d_min) / d_clear, d_min = the
           monitor's minimum robot <-> obstacle distance in the simulation (inf without obstacles).
close      on a close event: close_bonus x exp(-|v_gripper - v_object| / close_v_scale) for the first close with
           the gripper within the level's tolerances of the grasp pose; bad_close for every close outside them.
terminal   success / collision / dropped / timeout, as the harness decides them (success after HOLD_TIME).
Potentials are not zeroed at the end of an episode: Phi = 0 there would pay k_p x error for ending it early.
"""

from __future__ import annotations

import numpy as np

from dyngrasp.rl.observation import grasp_error


def grasp_errors(X_WG, X_WGgrasp) -> tuple[float, float]:
    """(position error in m, rotation error in rad) of the gripper w.r.t. the grasp pose."""
    e_p, e_r = grasp_error(X_WG, X_WGgrasp)
    return float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))


def progress_potential(X_WG, X_WGgrasp, in_hand: bool, rot_weight: float) -> float:
    if in_hand:
        return 0.0
    e_p, e_r = grasp_errors(X_WG, X_WGgrasp)
    return -(e_p + rot_weight * e_r)


def clearance_potential(d_min: float, d_clear: float, weight: float) -> float:
    return -weight * max(0.0, d_clear - d_min) / d_clear


def close_reward(good: bool, first_good: bool, v_rel: float, cfg: dict) -> float:
    """Reward of a close event: `good` = within the tolerances, `first_good` = no good close paid yet."""
    if not good:
        return cfg["bad_close"]
    return cfg["close_bonus"] * float(np.exp(-v_rel / cfg["close_v_scale"])) if first_good else 0.0
