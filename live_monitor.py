"""Headless check of a running robot runtime (no GUI): scan rate, health, and
how much the pose moves while the robot sits still, plus a PNG of the map.

Used for stationary-drift checks: keep the robot untouched while it runs.

    python live_monitor.py <pi-address> <seconds> <out.png>
    python live_monitor.py raspberrypi.local 60 /tmp/live_map.png
"""
import math
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from network.client import OperatorClient

HOST, DURATION, OUT = sys.argv[1], float(sys.argv[2]), sys.argv[3]

c = OperatorClient(HOST, 8765, client_name="live-monitor").start()
assert c.wait_for(lambda cl: cl.state is not None and cl.map is not None, timeout=15), "no state"
s0 = c.state
poses, healths, t_end = [], set(), time.monotonic() + DURATION
last_seq = None
while time.monotonic() < t_end:
    s = c.state
    if s["seq"] != last_seq:
        last_seq = s["seq"]
        p = s["pose"]
        poses.append((s["t"], p["x"], p["y"], p["theta"]))
        healths.add((s["health"]["state"], s["health"]["losses"]))
    time.sleep(0.05)
s1 = c.state
m = c.map
c.close()

a = np.array(poses)
dt = s1["t"] - s0["t"]
st0, st1 = s0["stats"], s1["stats"]
xy = a[:, 1:3]
print(f"window {dt:.1f} s, {len(a)} state samples")
print(f"scans: {(st1['scans_received'] - st0['scans_received']) / dt:.2f} Hz received, "
      f"dropped +{st1['scans_dropped'] - st0['scans_dropped']}, "
      f"accepted {st1['scans_accepted'] - st0['scans_accepted']}/{st1['scans_processed'] - st0['scans_processed']}")
print(f"health states seen: {sorted(healths)}; tick overruns total {st1['tick_overruns']}, max tick {st1['max_tick_ms']} ms")
print(f"pose start ({a[0,1]:.3f}, {a[0,2]:.3f}, {math.degrees(a[0,3]):.1f} deg)  end ({a[-1,1]:.3f}, {a[-1,2]:.3f}, "
      f"{math.degrees(a[-1,3]):.1f} deg)")
print(f"net drift over window: {math.hypot(*(xy[-1] - xy[0]))*100:.1f} cm, "
      f"{math.degrees(math.remainder(a[-1,3] - a[0,3], math.tau)):.2f} deg; "
      f"spread (std) x {xy[:,0].std()*100:.2f} cm, y {xy[:,1].std()*100:.2f} cm, "
      f"theta {math.degrees(a[:,3].std()):.2f} deg; max step {np.abs(np.diff(xy, axis=0)).max()*100:.2f} cm")
ranges = np.array(s1["scan"]["ranges"])
valid = ranges < s1["scan"]["range_max"]
print(f"latest scan: {valid.sum()}/{len(ranges)} valid bins, range {ranges[valid].min():.2f}-{ranges[valid].max():.2f} m")

# map snapshot: occupancy (row 0 = lowest y -> flip), robot pose, current scan hits
cells = m["cells"]
res, ox, oy = m["resolution"], m["origin_x"], m["origin_y"]
img = np.full(cells.shape, 205, np.uint8)
img[cells == 0] = 255
img[cells > 50] = 0
known = np.argwhere(cells != -1)
if len(known):
    (r0, c0), (r1, c1) = known.min(0) - 20, known.max(0) + 20
else:
    r0, c0, r1, c1 = 0, 0, *cells.shape
r0, c0 = max(r0, 0), max(c0, 0)
crop = np.flipud(img[r0:r1, c0:c1])
k = 6
im = Image.fromarray(crop).convert("RGB").resize((crop.shape[1] * k, crop.shape[0] * k), Image.NEAREST)
d = ImageDraw.Draw(im)
H = crop.shape[0]
def to_px(x, y):
    return ((x - ox) / res - c0) * k, (H - ((y - oy) / res - r0)) * k
sp = s1["scan_pose"]
ang = s1["scan"]["angle_min"] + np.arange(len(ranges)) * s1["scan"]["angle_increment"]
for r, th in zip(ranges[valid], ang[valid]):
    wx = sp["x"] + r * math.cos(sp["theta"] + th)
    wy = sp["y"] + r * math.sin(sp["theta"] + th)
    px, py = to_px(wx, wy)
    d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(220, 30, 30))
d.line([to_px(x, y) for x, y in xy], fill=(40, 120, 255), width=2)
px, py = to_px(sp["x"], sp["y"])
d.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(0, 90, 255), width=3)
d.line([px, py, px + 30 * math.cos(sp["theta"]), py - 30 * math.sin(sp["theta"])], fill=(0, 90, 255), width=3)
ox_px, oy_px = to_px(0, 0)
d.line([ox_px - 8, oy_px, ox_px + 8, oy_px], fill=(0, 160, 0), width=2)
d.line([ox_px, oy_px - 8, ox_px, oy_px + 8], fill=(0, 160, 0), width=2)
im.save(OUT)
print(f"map snapshot ({cells.shape}, {(cells > 50).sum()} occupied cells) -> {OUT}")
