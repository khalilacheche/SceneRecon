## Repo description

main.py: main file to run the training and inference
data.py: data loading and preprocessing (eval and train dataloaders, getting scans, etc.)
eval_2d.py: script to perform 2d evaluation (wrapper script to call the evaluation code provided by the original authors of the paper)
eval_3d.py: script to perform 3d evaluation (wrapper script to call the evaluation code provided by the original authors of the paper)
utils.py: utility functions
modules.py: implementation of the modules used in the model
fine_recon.py: main file for the model
collect_results.py: script to collect the results (json file for the 2d and 3d metrics for each scene) of the evaluation and save them in a csv file
keyframes.json: json file containing the keyframes for each scene
predict.sh: bash script to run the inference on all the scenes in the test set
configs: folder containing the config files for the experiments
evaluation: folder containing the source code for evaluation scripts
debug_scripts: folder containing scripts used for debugging, used locally


## Setup

### Environments
In the folder under `/scratch/students/2023-fall-acheche/conda/` you can find the conda environment for this project.
The environments used for this project are:
- scenerecon: used for training and inference
- eval_3d used for the 3d evaluation
- eval_2d used for the 2d evaluation

### Data

The ScanNet dataset with the finerecon format is located in 
`/scratch/students/2023-fall-acheche/data/scannet/scannet-finerecon/`

The depth estimates are located in
`/scratch/students/2023-fall-acheche/data/scannet/simplerecon_depth_est/`



## Training
To perform training, a config file must be created under the `configs` folder. The name of the config file refers to the `run_name` of the experiment.

To train a model, run with the `scenerecon` environment activated:

`python train.py --run_name the_name_of_the_config_file`

The model checkpoints will be saved in the `save_dir/the_name_of_the_config_file/SceneRecon/wandb_run_id/checkpoints`
specified in the config file.

You can also specify the `--resume` flag to resume training from a checkpoint. The checkpoint used is located in the `save_dir/the_name_of_the_config_file/model.ckpt` folder.

## Inference
To perform predictions on the project, you first have to put a checkpoint in the `save_dir/the_name_of_the_config_file/model.ckpt` folder.

Then, you can use 

`python train.py --run_name the_name_of_the_config_file --task predict`


This will run the inference on the test set (the scenes are defined in `/scratch/students/2023-fall-acheche/data/scannet/scannet-finerecon/scans/test.txt`), 


The results are saved in the `save_dir/the_name_of_the_config_file/outputs` specified in the config file.

You can also specify the `--scene_id sceneid` parameter to run the inference on a specific scene.

There is also a bash script `predict.sh` that runs the inference on all the scenes in the test set, but running the above command one at a time (to generate scenes one by one, in case there are some scenes that generate errors, like CUDA out of memory).


## Evaluation

To perform evaluation, there are two scripts: `eval_2d.py` and `eval_3d.py`.

### 3D evaluation

To perform the 3d evaluation, run with the `eval_3d` environment activated:

`python eval_3d.py --run_name the_name_of_the_config_file`

The results are saved in the `save_dir/the_name_of_the_config_file/outputs` specified in the config file. Each scene has a `sceneid_metrics_2d.json` file with the results.

3D evaluation is based on Transformer Fusion code provided by the original authors of the paper. The code for evaluation is located in the `evaluation/eval_3d` folder.

### 2D evaluation

For the 2d evaluation, run with the `eval_2d` environment activated:

`python eval_2d.py --run_name the_name_of_the_config_file`


Note: The evaluation uses pyrenderer and since we are on a server, we need to install OSMesa (following this tutorial https://pyrender.readthedocs.io/en/latest/install/index.html#installmesa). 



