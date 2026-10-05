"""Wheels-off-the-ground bench test for the motors (run ON THE PI first).

Put the robot on blocks so NO wheel touches anything, keep a hand on the
motor-battery switch / e-stop, then:

    python motor_test.py              # interactive: each step waits for Enter
    python motor_test.py --dry-run    # anywhere: prints duties, touches no GPIO

Steps (each prints what you should see):
    1. each of the 4 motors alone, forward then backward -- confirms the pin
       map in config/hardware.yaml (motors.pins) wheel by wheel
    2. body commands through the real driver (GpioDiffDriveRobot):
       forward, backward, turn LEFT (+angular: left wheels back, right
       wheels forward -- on the floor this turns counter-clockwise, which is
       the navigation stack's convention), turn RIGHT
    3. watchdog: one command, then silence -> must stop by itself after
       robot.yaml limits.command_timeout
If a wheel spins the wrong way: set motors.invert_left / invert_right (a whole
side) or swap that motor's two pins. If one side is stronger: trim it down.

Minimum-duty calibration (ON THE FLOOR, clear space, battery switch in hand):

    python motor_test.py --calibrate-duty forward
    python motor_test.py --calibrate-duty reverse
    python motor_test.py --calibrate-duty turn        # turn in place (left)
    python motor_test.py --calibrate-duty turn --duties 0.4,0.45,0.5,0.55,0.6

Steps through increasing duties (default 0.20 ... 0.65), 1.5 s each, asking
after each whether the ROBOT (not just the wheels) moved; the first "yes" is
the stall floor. Put it (plus ~0.05 margin) into config/hardware.yaml
motors: min_duty / min_reverse_duty / min_turn_duty. Drives the motors
directly (below the configured floors, on purpose); never above max_duty.
"""

from __future__ import annotations

import argparse
import time

from common.config import load_hardware_config, load_robot_config
from common.messages import VelocityCommand
from hardware.motor_driver import GpioDiffDriveRobot


def pause(prompt: str, auto: bool) -> None:
    print(f"\n>>> {prompt}")
    if auto:
        return
    input("    [Enter] to run this step, Ctrl+C to abort ")


def run_for(robot: GpioDiffDriveRobot, cmd: VelocityCommand, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        robot.set_velocity(cmd)          # refresh faster than the watchdog timeout
        time.sleep(0.05)
    print(f"    duty left={robot.last_duty[0]:+.2f} right={robot.last_duty[1]:+.2f}  motors={robot.motor_values()}")
    robot.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor bench test (robot on blocks).")
    parser.add_argument("--dry-run", action="store_true", help="touch no GPIO; print duties")
    parser.add_argument("--duty", type=float, default=0.45, help="duty for the single-motor step")
    parser.add_argument("--speed", type=float, default=0.15, help="m/s for the body-command step")
    parser.add_argument("--turn", type=float, default=0.8, help="rad/s for the turn step")
    parser.add_argument("--yes", action="store_true", help="do not wait for Enter between steps")
    parser.add_argument("--calibrate-duty", choices=("forward", "reverse", "turn"), default=None,
                        help="ON THE FLOOR: find the lowest duty that moves the robot this way")
    parser.add_argument("--duties", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65",
                        help="comma-separated duties for --calibrate-duty")
    args = parser.parse_args()
    if args.calibrate_duty:
        calibrate(args.calibrate_duty, [float(d) for d in args.duties.split(",")], args.dry_run)
        return

    motors_cfg, limits = load_hardware_config().motors, load_robot_config().limits
    robot = GpioDiffDriveRobot(motors_cfg, limits, dry_run=args.dry_run)
    auto = args.yes or robot.dry_run
    print(f"Motors: {'DRY RUN (no GPIO)' if robot.dry_run else 'REAL GPIO'}; pins {motors_cfg.pins}")
    try:
        for name, motor in robot._motors.items():
            pause(f"{name.replace('_', ' ').upper()} only: forward 1 s, then backward 1 s at duty {args.duty}", auto)
            motor.forward(args.duty)
            time.sleep(1.0)
            motor.backward(args.duty)
            time.sleep(1.0)
            motor.stop()

        steps = [
            ("FORWARD: all four wheels turn so the robot would drive forward", VelocityCommand(args.speed, 0.0)),
            ("BACKWARD: all four wheels reverse", VelocityCommand(-args.speed, 0.0)),
            ("TURN LEFT (+angular): LEFT wheels backward, RIGHT wheels forward", VelocityCommand(0.0, args.turn)),
            ("TURN RIGHT (-angular): LEFT wheels forward, RIGHT wheels backward", VelocityCommand(0.0, -args.turn)),
        ]
        for text, cmd in steps:
            pause(f"{text}  (cmd linear={cmd.linear} m/s, angular={cmd.angular} rad/s, 1.5 s)", auto)
            run_for(robot, cmd, 1.5)

        pause(f"WATCHDOG: one forward command, then silence -- wheels must stop by themselves "
              f"after {limits.command_timeout} s", auto)
        robot.set_velocity(VelocityCommand(args.speed, 0.0))
        t0 = time.monotonic()
        while robot.last_duty != (0.0, 0.0) and time.monotonic() - t0 < limits.command_timeout + 2.0:
            time.sleep(0.02)
        stopped_after = time.monotonic() - t0
        ok = robot.last_duty == (0.0, 0.0)
        print(f"    watchdog {'STOPPED the motors' if ok else 'DID NOT STOP THE MOTORS'} after {stopped_after:.2f} s")
    finally:
        robot.close()
        print("\nMotors stopped and released.")


def calibrate(motion: str, duties: list[float], dry_run: bool) -> None:
    motors_cfg, limits = load_hardware_config().motors, load_robot_config().limits
    duties = [d for d in duties if 0.0 < d <= motors_cfg.max_duty]
    robot = GpioDiffDriveRobot(motors_cfg, limits, dry_run=dry_run)
    key = {"forward": "min_duty", "reverse": "min_reverse_duty", "turn": "min_turn_duty"}[motion]
    # (left, right) wheel direction per motion; turn = LEFT turn (left back, right forward).
    sign = {"forward": (1, 1), "reverse": (-1, -1), "turn": (-1, 1)}[motion]
    side_of = {"front_left": 0, "back_left": 0, "front_right": 1, "back_right": 1}
    invert = (motors_cfg.invert_left, motors_cfg.invert_right)
    trim = (motors_cfg.left_trim, motors_cfg.right_trim)
    print(f"Calibrating {key} ({motion}); current value {getattr(motors_cfg, key):.2f}. "
          f"Robot ON THE FLOOR, clear space around it, battery switch in hand.")
    found = None
    try:
        for duty in duties:
            input(f"\n>>> duty {duty:.2f}: [Enter] to drive 1.5 s, Ctrl+C to abort ")
            for name, motor in robot._motors.items():
                side = side_of[name]
                d = duty * trim[side] * sign[side] * (-1 if invert[side] else 1)
                (motor.forward if d > 0 else motor.backward)(abs(d))
            time.sleep(1.5)
            for motor in robot._motors.values():
                motor.stop()
            if input("    did the ROBOT move (not just the wheels twitch)? [y/N] ").strip().lower().startswith("y"):
                found = duty
                break
    finally:
        robot.close()
    if found is None:
        print(f"\nNo duty up to {max(duties):.2f} moved the robot ({motion}). Check wiring/battery before raising max_duty.")
    else:
        print(f"\nStall floor for {motion}: {found:.2f}. Suggested config/hardware.yaml motors: "
              f"{key}: {min(motors_cfg.max_duty, found + 0.05):.2f}")


if __name__ == "__main__":
    main()
