import os
import subprocess

SCRIPT_PATH = ".artifact/scripts/workdir"

scripts = [x for x in os.listdir(SCRIPT_PATH) if x.endswith(".slurm")]
previous_id = 0

for idx, script in enumerate(scripts):
    print(f"Previous ID: {previous_id}")
    if previous_id == 0:
        job = f"sbatch {os.path.join(SCRIPT_PATH, script)}"
    else:
        job = f"sbatch --dependency=afterany:{previous_id} {os.path.join(SCRIPT_PATH, script)}"
    # previous_id = subprocess.check_output(job, shell=True, encoding="UTF-8")
    # previous_id = previous_id.replace("Submitted batch job ", "")
    # previous_id = previous_id.replace("\n", "")
