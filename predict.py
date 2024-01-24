import subprocess
import argparse
import os

SCANS_FILE = "/scratch/students/2023-fall-acheche/data/scannet/scannet-finerecon/scans/test_orig.txt"


parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
)
args = parser.parse_args()

command = "source activate /scratch/students/2023-fall-acheche/conda/scenerecon && cd /scratch/students/2023-fall-acheche/SceneRecon && python main.py --run_name {run_name} --task predict --scene {scene}"
with open(os.path.join(SCANS_FILE), "r") as f:
    scenes = sorted(set(f.read().strip().split()))


# Iterate through the commands
for scene in scenes:
    try:
        c = command.format(run_name = args.r,scene=scene)
        subprocess.run(c, shell=True, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{command}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")