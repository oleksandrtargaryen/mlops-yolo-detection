from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True,
    # нова версія ultralytics кладе відносний project у runs/detect/, тому абсолютний шлях
    project=str(Path("runs").absolute()),
    name="smoke_predict"
)

print("Smoke test completed.")
print(f"Number of result objects: {len(results)}")
