import os

SCRIPT_PATH = ".artifact/scripts/pjlab/chains_uniform"

scripts = [x for x in os.listdir(SCRIPT_PATH) if x.endswith(".slurm")]

for script in scripts:
    job = f""