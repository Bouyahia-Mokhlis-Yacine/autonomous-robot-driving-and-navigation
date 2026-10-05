"""Operator station (laptop side): connect to a robot runtime and drive it.

Displays ONLY what the robot publishes (map, planning buffer, LiDAR, pose,
goal, path, navigation/localization/safety state) and sends commands (goal,
cancel, STOP, RESUME). It runs no SLAM or planning of its own. See
visualization/operator_gui.py and network/protocol.py.

Run (robot runtime first -- robot_runtime.py on the Pi, or locally):
    conda activate AI-env
    python operator_gui.py                        # robot on this machine (127.0.0.1:8765)
    python operator_gui.py --host 192.168.1.50    # robot = the Raspberry Pi's IP (robot-centred close-up)
    python operator_gui.py --host 192.168.1.50 --full-map   # the original whole-map view
    python operator_gui.py --host 192.168.1.50 --orientation-check
                                                  # close-up LiDAR direction check (display only, no goals)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import tkinter as tk

from network.client import OperatorClient
from common.config import load_robot_config
from visualization.operator_gui import CloseUpOperatorGUI, OperatorGUI, OrientationCheckGUI


def fit_scale(width_m: float, height_m: float) -> float:
    probe = tk.Tk()
    probe.withdraw()
    screen_w, screen_h = probe.winfo_screenwidth(), probe.winfo_screenheight()
    probe.destroy()
    return max(8.0, min((screen_w - 60) / width_m, (screen_h - 260) / height_m))


def main() -> None:
    parser = argparse.ArgumentParser(description="Operator GUI for the robot runtime.")
    parser.add_argument("--host", default="127.0.0.1", help="robot runtime address (the Pi's IP)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scale", type=float, default=None, help="pixels per metre (default: fit screen)")
    parser.add_argument("--wait", type=float, default=30.0, help="seconds to wait for the robot's first map")
    parser.add_argument("--orientation-check", action="store_true",
                        help="robot-centred close-up with FRONT/LEFT/RIGHT/REAR, live scan and nearest-object "
                             "readout, for checking the LiDAR direction by hand (display only; goals disabled)")
    parser.add_argument("--view-radius", type=float, default=None,
                        help="metres shown each side of the robot (default 3.0; 2.5 with --orientation-check)")
    parser.add_argument("--full-map", action="store_true",
                        help="the original whole-map view instead of the robot-centred close-up")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    client = OperatorClient(args.host, args.port).start()
    print(f"Connecting to robot at {args.host}:{args.port} ...", flush=True)
    end = time.monotonic() + args.wait
    last_status = None
    while client.map is None or client.state is None:
        if time.monotonic() > end:
            print(f"No map from the robot after {args.wait:.0f} s ({client.status}). Is robot_runtime.py running?")
            client.close()
            sys.exit(1)
        if client.status != last_status:
            print(f"  {client.status}", flush=True)
            last_status = client.status
        time.sleep(0.1)
    m = client.map
    print(f"Connected: backend={client.state['backend']}, map {m['width']}x{m['height']} cells @ {m['resolution']} m, "
          f"origin ({m['origin_x']}, {m['origin_y']})", flush=True)

    width_m, height_m = m["width"] * m["resolution"], m["height"] * m["resolution"]
    robot = load_robot_config().robot
    if args.orientation_check:
        radius = args.view_radius or 2.5
        gui = OrientationCheckGUI(client, robot_length=robot.length, robot_width=robot.width,
                                  half_width_m=radius, scale=args.scale or fit_scale(2 * radius, 2 * radius), dt=0.1,
                                  lidar_offset=(robot.lidar_x, robot.lidar_y))
    elif args.full_map:
        gui = OperatorGUI(client, scale=args.scale or fit_scale(width_m, height_m), dt=0.1)
    else:
        radius = args.view_radius or 3.0
        side_px = int(fit_scale(2 * radius, 2 * radius) * 2 * radius)
        gui = CloseUpOperatorGUI(client, robot_length=robot.length, robot_width=robot.width,
                                 view_radius_m=radius, side_px=side_px, dt=0.1)
    try:
        gui.run()
    finally:
        client.close()  # the robot's operator-loss policy takes it from here


if __name__ == "__main__":
    main()
