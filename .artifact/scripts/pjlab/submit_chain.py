import os
import subprocess
SCRIPT_PATH = ".artifact/scripts/pjlab/chains_uniform"

scripts = [x for x in os.listdir(SCRIPT_PATH) if x.endswith(".slurm")]

for idx, script in enumerate(scripts):
    previous_id = 0
    if idx == 0:
        job = f"sbatch {os.path.join(SCRIPT_PATH, script)}"
    else:
        job = f"sbatch --dependency=after:{previous_id}:+5 {os.path.join(SCRIPT_PATH, script)}"
    previous_id = subprocess.check_output(job, shell=True)
    previous_id = previous_id.replace("Submitted batch job", "")
    