from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(description="Run prediction using a trained YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--source", type=str, default="datasets/construction-ppe-small/images/test",
                        help="Path to image, directory, or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--project", type=str, default="runs/lab03_prediction",
                        help="Directory for prediction results")
    parser.add_argument("--name", type=str, default="predictions_best_model",
                        help="Prediction run name")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction source not found: {source_path}")

    model = YOLO(str(model_path))
    model.predict(
        source=str(source_path),
        conf=args.conf,
        project=str(Path(args.project).absolute()),
        name=args.name,
        save=True,
        exist_ok=True,
    )

    prediction_dir = Path(args.project) / args.name
    print("Prediction completed")
    print(f"Results saved to: {prediction_dir}")


if __name__ == "__main__":
    main()
