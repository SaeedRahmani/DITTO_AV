# 20 paired videos — best v0.2 model (DITTO-AV shaped @999)

10 dev-10 routes, ONE CARLA run each, recorded two ways from the SAME
run (frame-synchronized): `route<N>_3d.mp4` (chase camera) and
`route<N>_2d.mp4` (BEV rendered from the run's per-tick state log,
kept as `route<N>_state.jsonl` for restyled re-renders).

Locations: /scratch/$USER/ditto_av/outputs/videos20/ (primary),
~/ditto_out/videos20/ (wipe-safe backup). Producer: job 10577433
(scripts/slurm/v02_video20.sbatch, agent state dump + cv2 BEV renderer).

Run scores (1 rep each; all 100% completion):
17569:100  25378:100  25381:100  25424:100  26405:100  27494:100
3514:65  3255:60  28198:60  2091:48
