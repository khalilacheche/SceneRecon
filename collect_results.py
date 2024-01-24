import argparse
import os
import glob
import pandas as pd
import json


def main(prediction_dir):
    metrics_3d_pattern = os.path.join(prediction_dir,'scene0[0-9][0-9][0-9]_[0-9][0-9]_metrics_3d.json')
    metrics_2d_pattern = os.path.join(prediction_dir,'scene0[0-9][0-9][0-9]_[0-9][0-9]_metrics.json')
    metrics_3d_paths = glob.glob(metrics_3d_pattern)
    metrics_2d_paths = glob.glob(metrics_2d_pattern)
    
    data_3d = []
    for metrics_3d_path in metrics_3d_paths:
        with open(metrics_3d_path, 'r') as metrics_3d_file:
            json_data = json.load(metrics_3d_file)
            data_3d.append(json_data)
    df_3d_metrics = pd.DataFrame(data_3d)
    data_2d = []
    for metrics_2d_path in metrics_2d_paths:
        with open(metrics_2d_path, 'r') as metrics_2d_file:
            json_data = json.load(metrics_2d_file)
            json_data["scene_id"] = os.path.basename(metrics_2d_path).split("_metrics")[0]
            data_2d.append(json_data)
    df_2d_metrics = pd.DataFrame(data_2d)
    df_res = pd.merge(df_2d_metrics, df_3d_metrics, on='scene_id', how='outer',suffixes=('_2d', '_3d'))
    df_res.to_csv(os.path.join(prediction_dir,"results.csv"),index=False)
    return df_res
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--outputs_dir", default=None)

    args = parser.parse_args()

    if args.run_name:
        prediction_dir = os.path.join("./save_dir", args.run_name, "outputs")
    else:
        prediction_dir = args.outputs_dir

    main(prediction_dir)