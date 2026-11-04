"""Approach (B): space-time RRT-Connect planning for grasping a moving target.

st_rrt        generic (q, t) RRT-Connect, time-preserving shortcutting, escape planning
collision     SpaceTimeChecker: Drake collision queries at a hypothetical (q, t)
goals         intercept-time candidates, pre-grasp IK and the grasp "tail" (descend/close/lift)
controller    SpaceTimeRRTController: the Controller that plans, replans and executes
"""
