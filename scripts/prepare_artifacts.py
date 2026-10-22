from argparse import ArgumentParser
from pathlib import Path
import shutil


def parse_args():
    parser = ArgumentParser(description="Prepare YOLO training artifacts for DVC tracking")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to YOLO training run directory")
    parser.add_argument("--prediction-dir", type=str, required=True,
                        help="Path to YOLO prediction results directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/lab03",
                        help="Directory for prepared artifacts")
    parser.add_argument("--max-predictions", type=int, default=5,
                        help="Maximum number of prediction images to copy")
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Skipped missing file: {src}")


def copy_prediction_examples(prediction_dir: Path, output_dir: Path, max_predictions: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        list(prediction_dir.glob("*.jpg"))
        + list(prediction_dir.glob("*.jpeg"))
        + list(prediction_dir.glob("*.png"))
    )
    selected_images = image_paths[:max_predictions]
    for image_path in selected_images:
        copy_file(image_path, output_dir / image_path.name)
    print(f"Copied {len(selected_images)} prediction examples")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Training run directory not found: {run_dir}")
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    model_dir = output_dir / "model"
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    prediction_output_dir = output_dir / "predictions"
    metadata_dir = output_dir / "metadata"

    copy_file(run_dir / "weights" / "best.pt", model_dir / "best.pt")
    copy_file(run_dir / "weights" / "last.pt", model_dir / "last.pt")
    copy_file(run_dir / "results.csv", metrics_dir / "results.csv")
    copy_file(run_dir / "results.png", plots_dir / "results.png")
    copy_file(run_dir / "confusion_matrix.png", plots_dir / "confusion_matrix.png")
    copy_file(run_dir / "confusion_matrix_normalized.png", plots_dir / "confusion_matrix_normalized.png")
    copy_file(run_dir / "args.yaml", metadata_dir / "args.yaml")
    copy_file(run_dir / "run_metadata.json", metadata_dir / "run_metadata.json")

    copy_prediction_examples(prediction_dir, prediction_output_dir, args.max_predictions)

    print()
    print("Artifact preparation completed")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
