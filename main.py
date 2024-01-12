import argparse
import os
import shutil

import box
import pytorch_lightning as pl
import torch
import yaml

import data
import fine_recon
import utils



SAVE_DIR_BASE = "./save_dir"
CONFIGS_DIR = "./configs"
MODEL_SAVE_NAME = "model.ckpt"
def load_config(config_fname):
    with open(config_fname, "r") as f:
        config = box.Box(yaml.safe_load(f))

    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        config.accelerator = "gpu"
        config.n_devices = n_gpus
    else:
        config.accelerator = "cpu"
        config.n_devices = 1

    return config


@pl.utilities.rank_zero_only
def zip_code(save_dir,config_path):
    os.system(f"zip {save_dir}/code.zip *.py {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", default="train", choices=["train", "predict", "find_lr"]
    )
    parser.add_argument("--resume",action='store_true')
    parser.add_argument("--run_name")
    args = parser.parse_args()


    
    run_name = args.run_name
    config_path = os.path.join(CONFIGS_DIR,f"{run_name}.yml")
    save_dir = os.path.join(SAVE_DIR_BASE, run_name)
    os.makedirs(os.path.join(save_dir,"outputs"), exist_ok=True)
    ckpt_path = os.path.join(save_dir,MODEL_SAVE_NAME)
    ckpt_exists = os.path.isfile(ckpt_path)
    ckpt_path = ckpt_path if ckpt_exists else None

    config = load_config(config_path)

    if args.task == "predict":
        config.n_devices = 1
        config.offline_logging = True
        assert ckpt_path is not None, "Model weights not found"
    else:
        if not args.resume:
            ckpt_path = None

    if ckpt_path is not None:
        shutil.copy(ckpt_path, ckpt_path + ".bak")

    model = fine_recon.FineRecon(config)
    if config.offline_logging:
        logger = pl.loggers.WandbLogger(
            project="SceneRecon",
            name=run_name,
            save_dir=save_dir,
            offline = True
        )
    else:
        logger = pl.loggers.WandbLogger(
            entity="khalil-acheche",
            project="SceneRecon",
            name=run_name,
            save_dir=save_dir,
            log_model="all",
        )

    logger.experiment

    zip_code(logger.save_dir,config_path)

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.accelerator,
        devices=config.n_devices,
        max_steps=config.steps,
        log_every_n_steps=50,
        precision=16,
        #detect_anomaly=(run_name=="test"),
        #############
        #num_sanity_val_steps=0,
        #############
        strategy="ddp" if config.n_devices > 1 else None,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="loss_val/loss", save_top_k=10)
        ],
    )
    

    if args.task == "train":
        trainer.fit(model, ckpt_path=ckpt_path)

    elif args.task == "find_lr":
        tuner = pl.tuner.Tuner(trainer)
        model.lr = model.config.initial_lr
        lr_finder = tuner.lr_find(
            model, train_dataloaders=model.train_dataloader(), val_dataloaders=[]
        )
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr.png")

    elif args.task == "predict":
        trainer.predict(model, ckpt_path=ckpt_path)
    else:
        raise NotImplementedError
