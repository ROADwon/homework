"""객체 감지 및 추적: 학습된 모델에 ByteTrack 추적기를 붙여 결과 영상을 만든다.

사용법:
    python track.py --source videos/input.mp4
    python track.py --source videos/input.mp4 --start 12 --seconds 5
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
from ultralytics import YOLO

import tool

import time

def resolve_stream(url):
    """유튜브 URL이면 실제 스트림 주소로 바꾸고, 그 밖의 주소는 그대로 돌려준다."""
    if "youtube.com" not in url and "youtu.be" not in url:
        return url
    import yt_dlp
    options = {
        "format": "bv*[height<=720][vcodec^=avc1]/bv*[height<=720]/bv*",
        "js_runtimes": {"node": {}},
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)["url"]


def track_live(source, conf=0.25, skip=0):
    """실시간 스트림을 q 키를 누를 때까지 계속 추적해 화면에 보여준다.

    skip: 추적 사이에 건너뛸 프레임 수. PC가 느려 화면이 밀리면 1~3으로 올린다.
    """
    yolo = YOLO(str(tool.WEIGHTS))
    while True:
        cap = cv2.VideoCapture(resolve_stream(source))   # 재연결할 때마다 주소를 새로 받는다
        while cap.isOpened():
            for _ in range(skip):
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                break
            result = yolo.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)[0]
            cv2.imshow("ship tracking (q: quit)", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        print("스트림이 끊겼습니다. 5초 뒤 다시 연결합니다.")
        time.sleep(5)

def track(source, out=tool.RESULTS_DIR / "tracking.mp4", seconds=5.0, start=0.0, conf=0.25):
    """source 영상의 start초 지점부터 seconds초 분량을 추적해 out에 저장한다."""
    source, out = Path(source), Path(out)
    yolo = YOLO(str(tool.WEIGHTS))

    with tempfile.TemporaryDirectory() as tmp:
        # Windows의 OpenCV는 한글 경로의 영상을 읽고 쓰지 못하는 경우가 있어 임시 폴더를 거친다.
        tmp_in = Path(tmp) / ("input" + source.suffix)
        tmp_out = Path(tmp) / "output.mp4"
        shutil.copyfile(source, tmp_in)

        cap = cv2.VideoCapture(str(tmp_in))
        if not cap.isOpened():
            raise RuntimeError(f"입력 영상을 열 수 없습니다: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

        # GitHub 웹 화면에서 바로 재생되도록 H.264(avc1)를 먼저 시도하고, 안 되면 mp4v로 저장한다.
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                break

        track_ids = set()
        frames = 0
        while frames < int(fps * seconds):
            ok, frame = cap.read()
            if not ok:
                break
            # persist=True: 프레임 사이에 추적 ID를 유지
            result = yolo.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)[0]
            writer.write(result.plot())   # 박스, 부류 이름, 신뢰도, 추적 ID를 그린 프레임
            if result.boxes.id is not None:
                track_ids.update(result.boxes.id.int().tolist())
            frames += 1

        cap.release()
        writer.release()
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_out, out)

    print(f"결과 영상: {out} ({frames / fps:.1f}초, {frames}프레임, 코덱 {codec}, 추적 ID {len(track_ids)}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="선박 객체 감지 및 추적")
    parser.add_argument("--source", required=True, help="입력 영상 파일")
    parser.add_argument("--seconds", type=float, default=5.0, help="결과 영상 길이(초)")
    parser.add_argument("--start", type=float, default=0.0, help="시작 지점(초)")
    parser.add_argument("--conf", type=float, default=0.25, help="감지 신뢰도 임계값")
    parser.add_argument("--live", action="store_true", help="실시간 스트림을 계속 추적")
    parser.add_argument("--skip", type=int, default=0, help="건너뛸 프레임 수 (실시간 모드)")
    args = parser.parse_args()

    if args.live:
        track_live(args.source, conf=args.conf, skip=args.skip)
    else:
        track(args.source, seconds=args.seconds, start=args.start, conf=args.conf)


    track(args.source, seconds=args.seconds, start=args.start, conf=args.conf)