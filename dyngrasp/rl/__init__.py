"""Approach (D): SAC (sbx) on the evaluation's Drake simulation.

actions      policy action -> velocity ramp, joint position commands, gripper opening
observation  the flat observation vector (fixed scales)
rewards      progress / clearance potentials, close events, terminals
scenes       curriculum levels, validation scenes, disjointness from the evaluation sets
curriculum   level bookkeeping
env          GraspEnv (gymnasium), one 0.1 s policy step per `step`
controller   RLController: a checkpoint's actor as a Controller (evaluation)
config       every number, in one dict
train        SAC training loop: checkpoints, resume, validation, TensorBoard
"""
