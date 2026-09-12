#!/usr/bin/env bash
# THE CONTROL FOR THE ORACLE-FREE HUNT (#261).
#
# The animal reaches 30 mm from its food with the prey-position oracle off.
# The arena is small and the animal wanders, so that number means nothing on
# its own. This runs each seed twice -- once normally, once with every
# committed bearing replaced by a uniform random one and EVERYTHING ELSE
# identical: same search pattern, same freezes, same fixations, same decision
# rule, same seed, same number of commitments.
#
# If the scrambled animal gets just as close, the eye is not doing the work.
# Result at 8 seeds: seeing closer on 7 of 8, median 60.6 mm against 152.8 mm,
# exact paired permutation p = 0.0234. See artifacts/evidence/session12/.
OUT=artifacts/evidence/session12/blind_control.tsv
mkdir -p artifacts/evidence/session12
echo -e "condition\tseed\tclosest_mm\tdecisions\tright\ton_screen" > $OUT
for seed in 2 3 4 5 6 7 8 9; do
for mode in "" "--blind"; do
timeout 900 .venv/Scripts/python.exe tools/oracle_free_hunt_video.py --steps 3000 --seed $seed --no-video $mode 2>&1 | .venv/Scripts/python.exe -c "
import sys,json; t=sys.stdin.read()
try: d=json.loads(t)
except Exception: raise SystemExit
lab='SEES' if 'CONTROL' not in d['run'] else 'BLIND'
print(f\"{lab}\t{d['seed']}\t{d['closest_approach_mm']}\t{d['decisions']}\t{d['decisions_with_prey_really_on_screen']}\t{d['frames_prey_on_screen']}\")" >> $OUT
done; done
echo DONE >> $OUT
