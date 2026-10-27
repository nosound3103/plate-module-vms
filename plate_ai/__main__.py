"""Signal-aware CLI for the standalone camera-sync workflow."""
import logging
import signal
import threading
import time
from .config import Settings
from .service import CameraSyncService


def main():
    try:
        settings = Settings.from_env()
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
        with CameraSyncService(settings):
            while not stop.wait(.5):
                pass
    except Exception as error:
        logging.getLogger(__name__).error("Camera sync failed type=%s", type(error).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
