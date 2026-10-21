# MLOps YOLO Object Detection Project

## Опис

Навчальний MLOps-проєкт для задачі object detection з використанням YOLO.

## Production-сценарій

Автоматичне виявлення людей та засобів індивідуального захисту на будівельному майданчику.

## Основні інструменти

- Ubuntu Server
- Python
- Git / GitHub
- DVC
- Ultralytics YOLO
- Jupyter Notebook
- FastAPI
- Docker
- CI/CD
- Kubernetes
- Monitoring

## Структура проєкту

- `src/` — Python-код
- `data/` — дані
- `datasets/` — датасети
- `models/` — моделі
- `notebooks/` — Jupyter Notebook
- `configs/` — конфігурації
- `tests/` — тести
- `docker/` — Docker-файли
- `reports/` — звіти

## Smoke test YOLO

```bash
yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg' conf=0.25
```
