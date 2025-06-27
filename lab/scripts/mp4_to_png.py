import argparse
import subprocess
from pathlib import Path


def convert(video_path: Path, out_dir: Path, fps: int) -> None:
    """Convert a single mp4 file to a PNG frame sequence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-vsync",
        "0",
        str(out_dir / "%06d.png"),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert MP4 videos to PNG sequences")
    p.add_argument("--video-dir", type=str, required=True, help="Directory with mp4 files")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for frames")
    p.add_argument("--fps", type=int, default=1, help="Frames per second to extract")
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    out_root = Path(args.out_dir)
    for video_file in sorted(video_dir.glob("*.mp4")):
        convert(video_file, out_root / video_file.stem, args.fps)


if __name__ == "__main__":
    main()
