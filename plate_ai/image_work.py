"""One CPU-only, lower-priority process for evidence codecs and disk work."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os


def _initialize():
    # Spawn keeps the worker away from the parent's CUDA context and locks.
    import cv2
    cv2.setNumThreads(1)
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass


def image_executor():
    executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                    initializer=_initialize)
    try:
        # Pay spawn/import cost during service setup, not on the first crossing.
        executor.submit(os.getpid).result(timeout=15.)
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    return executor


def encode_snapshot(frame, quality, local_dir=None, annotation=None):
    import cv2
    if local_dir:
        from .snapshots import save_original
        try:
            save_original(frame, local_dir)
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning("Snapshot result=LOCAL_SAVE_FAILED type=%s", type(error).__name__)
    if annotation is not None:
        from .passage_evidence import annotate_passage
        frame = annotate_passage(frame, annotation)
    ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return image.tobytes()
