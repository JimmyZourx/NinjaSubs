import argparse
from pathlib import Path

from app.services.sync_service import _duration, _normalize


def inspect(path: Path):
    data = path.read_bytes()
    normalized, cues = _normalize(data)

    return {
        "path": path,
        "bytes": len(data),
        "normalized_bytes": len(normalized),
        "cues": len(cues),
        "duration": _duration(cues),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arabic", type=Path, required=True)
    parser.add_argument("--english", type=Path, required=True)
    args = parser.parse_args()

    ar = inspect(args.arabic)
    en = inspect(args.english)

    print("=== Arabic ===")
    print("File     :", ar["path"])
    print("Bytes    :", ar["bytes"])
    print("Cues     :", ar["cues"])
    print("Duration :", f'{ar["duration"] / 60000:.2f} min')

    print()

    print("=== English ===")
    print("File     :", en["path"])
    print("Bytes    :", en["bytes"])
    print("Cues     :", en["cues"])
    print("Duration :", f'{en["duration"] / 60000:.2f} min')


if __name__ == "__main__":
    main()
