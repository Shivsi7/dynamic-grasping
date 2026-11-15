"""Approach (D): a learned policy on the evaluation's Drake simulation; behaviour cloning of (C) (SAC: branch rl-sac).

actions      policy action -> velocity ramp, joint position commands, gripper opening
observation  the flat observation vector (fixed scales)
rewards      progress / clearance potentials, close events, terminals
scenes       curriculum levels, validation scenes, disjointness from the evaluation sets
curriculum   level bookkeeping
env          GraspEnv (gymnasium), one 0.1 s policy step per `step`
controller   RLController: a checkpoint's actor as a Controller (evaluation)
config       every number, in one dict
demos        behaviour-cloning demonstrations from (C): (D) observations and actions, open-loop replay
bc           behaviour cloning of (C): the SAC actor trained on the demonstrations
train        SAC from scratch (sbx), the curriculum, validation, checkpoints
agent        SAC/RLPD agent in JAX for fine-tuning the BC actor
vecenv       asynchronous environment processes
finetune     SAC fine-tuning of the BC policy: demonstrations in the batches, warm-up, background validation
"""
