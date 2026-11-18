"""학습 실행. 학습이 끝나면 테스트까지 이어서 수행한다.

사용법:
    python main.py                 # 기본값으로 학습 + 테스트
    python main.py --epochs 100    # 에폭 수 변경
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

import test
import tool


def train(model="yolov8n.pt", epochs=50, imgsz=640, batch=16):
    """사전학습 YOLO 가중치를 3개 선박 부류로 학습하고 best.pt를 weights/에 복사한다."""
    data_yaml = tool.write_data_yaml()

    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(tool.RUNS_DIR),
        name="train",
        exist_ok=True,
    )

    save_dir = Path(yolo.trainer.save_dir)
    tool.WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(save_dir / "weights" / "best.pt", tool.WEIGHTS)

    # 보고서용 학습 곡선과 혼동 행렬
    out_dir = tool.RESULTS_DIR / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.png", "results.csv", "confusion_matrix.png"):
        if (save_dir / name).exists():
            shutil.copyfile(save_dir / name, out_dir / name)

    print(f"학습 완료: {tool.WEIGHTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="선박 3개 부류 학습")
    parser.add_argument("--model", default="yolov8n.pt", help="시작 가중치")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    train(args.model, args.epochs, args.imgsz, args.batch)
    test.evaluate()