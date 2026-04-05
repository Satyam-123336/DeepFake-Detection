from src.behavioral.blink.blink_detector import BlinkDetector


def test_blink_detector_counts_one_blink() -> None:
    detector = BlinkDetector(ear_threshold=0.2, min_consecutive_frames=2)
    timeline = [
        (0.0, 0.3),
        (0.1, 0.18),
        (0.2, 0.17),
        (0.3, 0.31),
    ]
    result = detector.detect(timeline)
    assert result.features.blink_count == 1