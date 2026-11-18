"""
선박 3개 부류(컨테이너선, 요트, 어선) 이미지 수집 스크립트 (v2)

사용법:
    pip install icrawler
    python collect_ship_images.py

결과:
    dataset/container_ship/*.jpg
    dataset/yacht/*.jpg
    dataset/fishing_boat/*.jpg

v2 변경점:
    - 부류당 검색어를 12개로 늘림 (Bing은 검색어 하나당 받아지는 양이 적음)
    - 검색어마다 깨진 파일·중복을 정리하고, 정리 후 장수가 TARGET에 도달할 때까지 계속 수집
    - 최소 크기 조건 완화
    - 이미 받은 이미지는 그대로 두고 이어서 수집 (기존 dataset 폴더를 지우지 않아도 됨)
"""

import hashlib
import logging
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from PIL import Image

ROOT = Path("dataset")
TARGET = 150          # 부류당 목표 장수 (수동 검수 후 100장 이상 남기기 위한 여유분 포함)
PER_KEYWORD = 200     # 검색어 하나당 최대 요청 장수 (실제로는 이보다 적게 받아짐)

CLASSES = {
    "container_ship": [
        "container ship",
        "container ship at sea",
        "container vessel",
        "container ship aerial view",
        "container ship port",
        "loaded container ship sailing",
        "ultra large container vessel",
        "feeder container ship",
        "container ship side view",
        "containerschiff",
        "컨테이너선",
        "컨테이너선 항해",
    ],
    "yacht": [
        "yacht",
        "sailing yacht",
        "sailing yacht at sea",
        "motor yacht",
        "luxury yacht",
        "superyacht at sea",
        "sailboat sailing",
        "yacht marina",
        "yacht regatta",
        "catamaran yacht",
        "요트",
        "세일링 요트",
    ],
    "fishing_boat": [
        "fishing boat",
        "fishing boat at sea",
        "fishing vessel",
        "fishing trawler",
        "commercial fishing boat",
        "small fishing boat harbor",
        "fishing boat port",
        "shrimp boat",
        "squid fishing boat",
        "purse seiner vessel",
        "어선",
        "어선 항구",
    ],
}


def clean_class(folder: Path) -> int:
    """깨진 이미지와 내용이 같은 중복 이미지를 삭제하고 남은 장수를 돌려준다."""
    seen = set()
    count = 0
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            path.unlink()
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            path.unlink()
        else:
            seen.add(digest)
            count += 1
    return count


def crawl_class(name: str, keywords: list[str]) -> int:
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)

    n = clean_class(folder)
    for kw in keywords:
        if n >= TARGET:
            break
        print(f"[{name}] 현재 {n}장 / 검색어: {kw}")
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": str(folder)},
            log_level=logging.WARNING,
        )
        crawler.crawl(
            keyword=kw,
            max_num=PER_KEYWORD,
            min_size=(200, 200),
            file_idx_offset="auto",   # 기존 파일 번호 뒤에 이어서 저장
        )
        n = clean_class(folder)
    return n


if __name__ == "__main__":
    results = {cls: crawl_class(cls, kws) for cls, kws in CLASSES.items()}

    print("\n=== 수집 결과 (깨진 파일·중복 제거 후) ===")
    for cls, n in results.items():
        status = "OK" if n >= 100 else "부족 - 검색어를 추가해 다시 실행하세요"
        print(f"{cls:15s}: {n:4d}장  {status}")