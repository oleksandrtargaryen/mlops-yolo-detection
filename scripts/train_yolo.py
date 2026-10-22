from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import platform
import sys

from ultralytics import YOLO


def parse_args():
    parser = ArgumentParser(
        description="Automated YOLO training for Construction-PPE-small dataset"
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base YOLO model or path to model weights")
    parser.add_argument("--data", type=str,
                        default="datasets/construction-ppe-small/data.yaml",
                        help="Path to dataset YAML configuration")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device: cpu or GPU index, for example 0")
    parser.add_argument("--project", type=str, default="runs/lab03_script",
                        help="Directory for experiment results")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name. If not specified, a timestamp-based name is used")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return parser.parse_args()


def create_experiment_name(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = Path(model_name).stem
    return f"{clean_model_name}_construction_ppe_{timestamp}"


def save_run_metadata(run_dir: Path, args) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "run_dir": str(run_dir),
        "best_model_path": str(run_dir / "weights" / "best.pt"),
        "last_model_path": str(run_dir / "weights" / "last.pt"),
    }
    output_path = run_dir / "run_metadata.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML file not found: {data_path}")

    if args.name is None:
        args.name = create_experiment_name(args.model)

    print("Starting YOLO training")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Experiment name: {args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        # відносний project ultralytics 8.4 кладе в runs/detect/, тому absolute
        project=str(Path(args.project).absolute()),
        name=args.name,
        patience=args.patience,
    )

    run_dir = Path(model.trainer.save_dir)
    save_run_metadata(run_dir, args)

    print()
    print("Training completed")
    print(f"Results saved to: {run_dir}")
    print(f"Best model: {run_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {run_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
