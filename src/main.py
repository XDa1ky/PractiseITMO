import argparse
import yaml
import cv2

from .detector import PersonDetector
from .video_stream import VideoStream


def exp_smooth(prev: float, current: float, alpha: float) -> float:
    """Simple exponential smoothing."""
    return alpha * current + (1 - alpha) * prev


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg_path: str):
    cfg = load_cfg(cfg_path)

    detector = PersonDetector(
        model_name=cfg.get("model_name", "yolov5s"),
        classes=cfg.get("classes", [0]),
        conf_threshold=cfg.get("conf_threshold", 0.5)
    )

    stream = VideoStream(
        source=cfg.get("video_source", 0),
        window_name=cfg.get("window_name", "People Counter")
    )

    alpha = cfg.get("smoothing_alpha", 0.3)
    smooth_cnt = 0.0

    print("[INFO] Started. Press 'q' to exit.")
    try:
        while True:
            frame = stream.read()
            annotated, cnt = detector.detect(frame)

            smooth_cnt = exp_smooth(smooth_cnt, cnt, alpha)

            cv2.putText(
                annotated,
                f"Count: {cnt}  Smooth: {smooth_cnt:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            stream.show(annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time people counter with YOLOv5")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)
