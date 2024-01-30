import evaluation.eval_3d.eval_3d as eval_3d
import argparse
import os



def main(prediction_dir):
    
    metrics_3d = eval_3d.evaluate(prediction_dir,True)
    print(metrics_3d)


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