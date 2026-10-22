from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser(description="Analyze YOLO training metrics from results.csv")
    parser.add_argument("--results", type=str, required=True, help="Path to YOLO results.csv file")
    parser.add_argument("--output-dir", type=str, default="artifacts/lab03/metrics",
                        help="Directory for saving metric summaries and plots")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip() for column in df.columns]
    return df


def find_column(df: pd.DataFrame, required_parts: list[str]) -> str:
    for column in df.columns:
        normalized_column = column.lower()
        if all(part.lower() in normalized_column for part in required_parts):
            return column
    raise KeyError(
        f"Column with parts {required_parts} was not found. "
        f"Available columns: {list(df.columns)}"
    )


def build_metric_summary(df: pd.DataFrame) -> dict:
    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

    precision_col = find_column(df, ["precision"])
    recall_col = find_column(df, ["recall"])
    map50_95_col = find_column(df, ["map50-95"])
    map50_col = find_column(df, ["map50"])

    if map50_col == map50_95_col:
        map50_col = find_column(df, ["map50("])

    best_idx = df[map50_95_col].idxmax()
    last_idx = df.index[-1]

    best_row = df.loc[best_idx]
    last_row = df.loc[last_idx]

    summary = {
        "best_epoch_by_map50_95": int(best_row[epoch_col]),
        "last_epoch": int(last_row[epoch_col]),
        "best_metrics": {
            "precision": float(best_row[precision_col]),
            "recall": float(best_row[recall_col]),
            "mAP50": float(best_row[map50_col]),
            "mAP50_95": float(best_row[map50_95_col]),
        },
        "last_epoch_metrics": {
            "precision": float(last_row[precision_col]),
            "recall": float(last_row[recall_col]),
            "mAP50": float(last_row[map50_col]),
            "mAP50_95": float(last_row[map50_95_col]),
        },
        "columns": {
            "epoch": epoch_col,
            "precision": precision_col,
            "recall": recall_col,
            "mAP50": map50_col,
            "mAP50_95": map50_95_col,
        },
    }
    return summary


def save_summary(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "metrics_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    rows = [
        {
            "stage": "best_epoch_by_mAP50_95",
            "epoch": summary["best_epoch_by_map50_95"],
            **summary["best_metrics"],
        },
        {
            "stage": "last_epoch",
            "epoch": summary["last_epoch"],
            **summary["last_epoch_metrics"],
        },
    ]
    csv_path = output_dir / "metrics_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved metric summary to: {json_path}")
    print(f"Saved metric table to: {csv_path}")


def plot_main_metrics(df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_col = summary["columns"]["epoch"]
    precision_col = summary["columns"]["precision"]
    recall_col = summary["columns"]["recall"]
    map50_col = summary["columns"]["mAP50"]
    map50_95_col = summary["columns"]["mAP50_95"]

    plt.figure()
    plt.plot(df[epoch_col], df[precision_col], label="precision")
    plt.plot(df[epoch_col], df[recall_col], label="recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Precision and Recall during training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_plot.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df[epoch_col], df[map50_col], label="mAP50")
    plt.plot(df[epoch_col], df[map50_95_col], label="mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("mAP metrics during training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "map_plot.png", dpi=150)
    plt.close()

    print(f"Saved metric plots to: {output_dir}")


def main():
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found: {results_path}")

    df = pd.read_csv(results_path)
    df = normalize_columns(df)

    summary = build_metric_summary(df)
    save_summary(summary, output_dir)
    plot_main_metrics(df, summary, output_dir)

    print()
    print("Metric analysis completed")
    print(json.dumps(summary, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
