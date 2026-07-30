# Phase-0b speed audit (2026-07-30)

Expert (val npz, 39k frames): mean 4.57 m/s, moving-mean 6.70,
median 5.26, stopped (<0.5 m/s) 32% of frames.
Agent (BC, 220-run tick logs, 105k ticks): mean 0.87 m/s, moving-mean
5.16, median 0.02, stopped 84% of ticks.

Verdict: cruising speed is close (5.2 vs 6.7); the pathology is STOP
FREQUENCY (84% vs 32%) — the policy brakes/stalls constantly. This is
the min-speed-penalty source (2309 events on the 220) and consistent
with brake-output flicker around the 0.5 binarization threshold.
Probes: brake_threshold {0.7, 0.85} 3x3s. Structural fix: Phase-1
waypoint+PID (speed profile encoded in waypoint spacing).

## Probe verdict (2026-07-30 evening)

brake_threshold 0.7 -> 19.91/63.9%; 0.85 -> 20.97/62.3%; baseline 0.5
-> 22.12/65.8%. Both arms slightly worse (noise-level). **Brake
binarization is NOT the stop mechanism** — the policy genuinely
commands stops (confident brake / collapsed throttle). Deployment-lever
list stays fully dead; the stop-frequency fix must be structural:
Phase-1 waypoint+PID (speed profile from waypoint spacing), plus a
privileged route-PID reference agent (0d) to bound the controller.
