"""Signal-aware CLI for configuration synchronization and vehicle detection."""
import argparse
from contextlib import ExitStack
import logging
import signal
import threading
import time
from .config import Settings
from .service import CameraSyncService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sync", "detection"), default="sync")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        detection_settings = None
        recognition_settings = None
        if args.mode == "detection":
            from .detection_config import DetectionSettings
            from .recognition_config import RecognitionSettings
            detection_settings = DetectionSettings.from_env()
            recognition_settings = RecognitionSettings.from_env()
    except ValueError as error:
        # Settings errors name the field, never the supplied value.
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error("Configuration error: %s", error)
        return 2
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)sZ %(levelname)s %(name)s %(message)s")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        with ExitStack() as stack:
            sync = stack.enter_context(CameraSyncService(settings))
            if detection_settings is not None:
                from .runtime import DetectionRuntime
                detection = stack.enter_context(DetectionRuntime(sync, detection_settings))
                if recognition_settings.enabled:
                    from .recognition_runtime import RecognitionRuntime
                    stack.enter_context(RecognitionRuntime(sync, detection, recognition_settings))
            while not stop.wait(.5):
                pass
    except Exception as error:
        logging.getLogger(__name__).error("Service failed mode=%s type=%s", args.mode, type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
