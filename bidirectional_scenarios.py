"""Bidirectional-motion scenarios (FORWARD / REVERSE / ROTATE_IN_PLACE) in the
closed-loop simulator, with the realistic LiDAR model by default.

    conda activate AI-env
    python bidirectional_scenarios.py                   # all scenarios, 5 seeds, + forward-only comparison
    python bidirectional_scenarios.py A_goal_behind --seeds 3
    python bidirectional_scenarios.py --sensor ideal

Per run: goal reached, final position error (m), final heading error (the
localization heading error at the end -- goals have no heading), collision,
minimum clearance between the robot footprint and any obstacle/wall (m),
forward<->reverse transitions, turns in place, safety stops, time.

NOTE: the simulator plans with robot.yaml's FULL inflation (0.57 m); the live
runtime uses runtime.yaml's 0.30 m. Rooms here are sized for 0.57 m.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from common.config import load_navigation_config, load_robot_config
from common.geometry import offset_pose, wrap_angle
from common.messages import Goal, Pose2D
from control.path_follower import MotionMode
from simulation.closed_loop import ClosedLoopSimulation
from simulation.environment import Environment
from simulation.obstacles import RectangleObstacle
from simulation.sensor_simulation_config import load_sensor_simulation_config

ROBOT = load_robot_config()
NAV = load_navigation_config()
LEN, WID = ROBOT.robot.length, ROBOT.robot.width


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[], tuple[Environment, Pose2D, Goal]]   # goal in the SLAM frame (= the LiDAR's frame at start)
    expect_goal: bool = True
    duration: float = 60.0


def room(w: float, h: float, start: Pose2D, boxes=()) -> Environment:
    env = Environment(width=w, height=h, robot_length=LEN, robot_width=WID, robot_pose=start)
    for b in boxes:
        env.add_static_obstacle(RectangleObstacle(*b))
    return env


def s_goal_behind():
    start = Pose2D(2.5, 2.0, 0.0)
    return room(5.0, 4.0, start), start, Goal(-1.0, 0.0)


def s_goal_side():
    start = Pose2D(2.5, 2.0, 0.0)
    return room(5.0, 4.0, start), start, Goal(0.0, 1.0)


def s_narrow_far_behind():
    # 1.6 m wide corridor-room, facing its end wall, goal 2 m behind (> max_reverse_distance).
    start = Pose2D(3.0, 0.8, 0.0)
    return room(4.2, 1.6, start), start, Goal(-2.0, 0.0)


def s_narrow_near_behind():
    start = Pose2D(3.0, 0.8, 0.0)
    return room(4.2, 1.6, start), start, Goal(-1.2, 0.0)


def s_obstacle_behind():
    # Box 0.15 m behind the rear edge; goal behind the box.
    start = Pose2D(2.5, 2.0, 0.0)
    box = (2.5 - LEN / 2 - 0.15 - 0.15, 2.0, 0.3, 0.6)
    return room(5.0, 4.0, start, [box]), start, Goal(-1.5, 0.0)


def s_obstacle_ahead():
    # Box 0.20 m ahead of the front edge; goal behind (free).
    start = Pose2D(2.5, 2.0, 0.0)
    box = (2.5 + LEN / 2 + 0.20 + 0.15, 2.0, 0.3, 0.6)
    return room(5.0, 4.0, start, [box]), start, Goal(-1.0, 0.0)


def s_boxed_in():
    # Boxes 0.15 m ahead and behind, side walls 0.10 m away: every motion blocked.
    start = Pose2D(2.0, 0.3, 0.0)
    boxes = [(2.0 + LEN / 2 + 0.15 + 0.15, 0.3, 0.3, 0.6), (2.0 - LEN / 2 - 0.15 - 0.15, 0.3, 0.3, 0.6)]
    return room(4.0, 0.6, start, boxes), start, Goal(1.2, 0.0)


def s_goal_120(sign: float):
    # Robot 2026-09-30: a goal ~120 deg off, farther than max_reverse_distance,
    # made the robot turn left/right forever.
    def build():
        start = Pose2D(3.0, 2.25, 0.0)   # centred: both goals stay outside the 0.57 m wall buffer
        a = math.radians(120.0 * sign)
        return room(5.0, 4.5, start), start, Goal(1.8 * math.cos(a), 1.8 * math.sin(a))
    return build


SCENARIOS = [
    Scenario("A_goal_behind", "goal 1 m straight behind", s_goal_behind),
    Scenario("B_goal_side", "goal 1 m to the left (90 deg)", s_goal_side),
    Scenario("C_narrow_far_behind", "1.6 m-wide room, facing the end wall, goal 2 m behind", s_narrow_far_behind),
    Scenario("C2_narrow_near_behind", "1.6 m-wide room, goal 1.2 m behind", s_narrow_near_behind),
    Scenario("D_obstacle_behind", "box 0.15 m behind the robot, goal behind the box", s_obstacle_behind),
    Scenario("E_obstacle_ahead", "box 0.20 m ahead, goal 1 m behind (free)", s_obstacle_ahead),
    Scenario("G_goal_120_left", "goal 1.8 m away, 120 deg to the left (was a left-right loop)", s_goal_120(1.0)),
    Scenario("G2_goal_120_right", "goal 1.8 m away, 120 deg to the right", s_goal_120(-1.0)),
    Scenario("F_boxed_in", "boxes 0.15 m ahead/behind, walls 0.10 m beside: nothing may move", s_boxed_in,
             expect_goal=False, duration=20.0),
]


def footprint_clearance(env: Environment) -> float:
    corners = env.robot_footprint_corners()
    pts = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        for t in np.linspace(0.0, 1.0, 26):
            pts.append((ax + t * (bx - ax), ay + t * (by - ay)))
    pts = np.array(pts)
    best = float(np.min(np.minimum.reduce([pts[:, 0], env.width - pts[:, 0], pts[:, 1], env.height - pts[:, 1]])))
    for o in env.all_obstacles():
        dx = np.maximum(np.abs(pts[:, 0] - o.x) - o.width / 2, 0.0)
        dy = np.maximum(np.abs(pts[:, 1] - o.y) - o.height / 2, 0.0)
        best = min(best, float(np.min(np.hypot(dx, dy))))
    return best


def run(scn: Scenario, seed: int, sensor: str, robot_cfg=ROBOT) -> dict:
    env, start, goal = scn.build()
    sim = ClosedLoopSimulation(environment=env, ground_truth_start=start, goal=goal, nav_cfg=NAV, robot_cfg=robot_cfg,
                               dt=0.1, max_simulation_time=scn.duration,
                               sensor_simulation=load_sensor_simulation_config(mode=sensor, seed=seed))
    min_clear = footprint_clearance(env)
    while sim._sim_time < scn.duration and not sim.metrics.collision and not sim.metrics.goal_reached:
        sim.step()
        min_clear = min(min_clear, footprint_clearance(env))
    m = sim.metrics
    follower = sim.nav_manager._path_follower
    entered = [b for _, b in follower.mode_changes]
    moving = [x for x in entered if x in (MotionMode.FORWARD, MotionMode.REVERSE)]
    fr_transitions = sum(1 for a, b in zip(moving, moving[1:]) if a is not b)
    gt = env.robot_pose
    origin = offset_pose(start, robot_cfg.robot.lidar_x, robot_cfg.robot.lidar_y)   # SLAM frame = LiDAR at start
    goal_world = (origin.x + goal.x * math.cos(origin.theta) - goal.y * math.sin(origin.theta),
                  origin.y + goal.x * math.sin(origin.theta) + goal.y * math.cos(origin.theta))
    return dict(goal=m.goal_reached, collision=m.collision, goal_err=math.hypot(gt.x - goal_world[0], gt.y - goal_world[1]),
                heading_err=abs(math.degrees(m.final_heading_error or 0.0)), min_clear=min_clear,
                fr_transitions=fr_transitions, rotations=entered.count(MotionMode.ROTATE_IN_PLACE),
                reverses=entered.count(MotionMode.REVERSE), safety_stops=sim.nav_manager.safety_stops,
                time=m.simulation_time, dist=m.ground_truth_distance, state=m.final_nav_state)


def legacy(robot_cfg):
    return dataclasses.replace(robot_cfg, control=dataclasses.replace(
        robot_cfg.control, turn_in_place_enter_deg=None, allow_reverse=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sensor", choices=("ideal", "realistic"), default="realistic")
    ap.add_argument("--no-compare", action="store_true", help="skip the forward-only (legacy) comparison")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    chosen = [s for s in SCENARIOS if not args.names or s.name in args.names]
    variants = [("bidirectional", ROBOT)] + ([] if args.no_compare else [("forward-only", legacy(ROBOT))])
    print(f"{'scenario':22s} {'controller':13s} {'goal':>5s} {'coll':>4s} {'goalErr':>7s} {'hdgErr':>6s} {'minClr':>6s} "
          f"{'F/R sw':>6s} {'rot':>4s} {'rev':>4s} {'stops':>5s} {'time':>5s} {'dist':>5s}  final states")
    for scn in chosen:
        for label, rcfg in variants:
            rs = [run(scn, seed, args.sensor, rcfg) for seed in range(args.seeds)]
            n = len(rs)
            mean = lambda k: sum(r[k] for r in rs) / n  # noqa: E731
            print(f"{scn.name:22s} {label:13s} {sum(r['goal'] for r in rs):>2d}/{n:<2d} {sum(r['collision'] for r in rs):>4d} "
                  f"{max(r['goal_err'] for r in rs):7.3f} {max(r['heading_err'] for r in rs):6.1f} "
                  f"{min(r['min_clear'] for r in rs):6.3f} {mean('fr_transitions'):6.1f} {mean('rotations'):4.1f} "
                  f"{mean('reverses'):4.1f} {mean('safety_stops'):5.1f} {mean('time'):5.1f} {mean('dist'):5.2f}  "
                  f"{sorted(set(r['state'] for r in rs))}", flush=True)
        print(f"{'':22s} ({scn.description}; expected: {'reach the goal' if scn.expect_goal else 'stay stopped'})")


if __name__ == "__main__":
    main()
