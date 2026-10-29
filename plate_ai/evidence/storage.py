"""Encode original frames and upload multipart images off the camera thread."""
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
from pathlib import Path, PurePosixPath
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import cv2
from plate_ai.evidence.worker import image_executor, encode_snapshot

LOGGER = logging.getLogger(__name__)


def save_original(frame, directory):
    destination = Path(directory).resolve()/f"{uuid4().hex}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise OSError("Could not save original crossing evidence")
    return str(destination)


def valid_object_key(value):
    return (isinstance(value, str) and value.startswith("plate/")
            and len(value) <= 2048 and not any(c in value for c in "\\?#%\r\n\x00:")
            and all(part not in {"", ".", ".."} for part in value.split("/"))
            and PurePosixPath(value).suffix.lower() in {".jpg", ".jpeg", ".png"})


class StorageClient:
    def __init__(self, host, settings, *, opener=urlopen):
        self.endpoint = f"https://{host}/storage/upload/file"
        self.settings, self.opener = settings, opener

    def upload(self, camera, image, *, object_key):
        if not image or len(image) > self.settings.max_bytes:
            LOGGER.error("Snapshot camera=%s event=%s result=INVALID_IMAGE_SIZE", camera.code, object_key)
            return None
        boundary = f"plate-{uuid4().hex}"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"camera_id\"\r\n\r\n"
                f"{camera.id}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"category\"\r\n\r\n"
                f"plate\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"snapshot.jpg\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n").encode()+image+f"\r\n--{boundary}--\r\n".encode()
        for attempt in range(self.settings.attempts):
            request = Request(self.endpoint, data=body, method="POST", headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"})
            try:
                # HTTPS certificate verification stays enabled. MQTT credentials
                # are not HTTP credentials and are never forwarded to Storage.
                with self.opener(request, timeout=self.settings.timeout_seconds) as response:
                    raw = response.read(65537)
                    if response.status != 200 or len(raw) > 65536:
                        raise ValueError("Invalid upload response")
                    result = json.loads(raw)
                if not isinstance(result, dict) or result.get("success") is not True or not valid_object_key(result.get("object_key")):
                    raise ValueError("Storage did not return a valid object_key")
                LOGGER.info("Snapshot camera=%s event=%s result=UPLOADED bytes=%d", camera.code, object_key, len(image))
                return result["object_key"]  # Never use the expiring/internal `url`.
            except HTTPError as error:
                retry = error.code == 429 or error.code >= 500
                LOGGER.warning("Snapshot camera=%s event=%s result=HTTP_ERROR status=%d attempt=%d",
                               camera.code, object_key, error.code, attempt+1)
                error.close()
                if not retry:
                    break
            except (URLError, TimeoutError, OSError) as error:
                LOGGER.warning("Snapshot camera=%s event=%s result=NETWORK_ERROR type=%s attempt=%d",
                               camera.code, object_key, type(error).__name__, attempt+1)
            except (ValueError, TypeError):
                LOGGER.error("Snapshot camera=%s event=%s result=INVALID_RESPONSE", camera.code, object_key)
                break
            if attempt+1 < self.settings.attempts:
                time.sleep(self.settings.retry_seconds)
        LOGGER.warning("Snapshot camera=%s event=%s result=PUBLISH_WITHOUT_IMAGE", camera.code, object_key)
        return None


class SnapshotService:
    def __init__(self, host, settings, *, client=None):
        self.settings = settings
        self.client = client if client is not None else StorageClient(host, settings)
        self._slots = threading.BoundedSemaphore(settings.max_pending)
        self._executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="plate-snapshot")
        self.image_executor = image_executor()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()

    def submit(self, camera, frame, *, object_key, local_dir=None, annotation=None):
        if frame is None or not self._slots.acquire(blocking=False):
            LOGGER.warning("Snapshot camera=%s event=%s result=%s", camera.code, object_key,
                           "NO_MATCHING_FRAME" if frame is None else "QUEUE_FULL")
            future = Future()
            future.set_result(None)
            return future
        try:
            future = self._executor.submit(self._prepare, camera, frame, object_key, local_dir, time.monotonic(), annotation)
        except Exception:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future

    def _prepare(self, camera, frame, object_key, local_dir, created, annotation):
        if time.monotonic()-created >= self.settings.wait_seconds:
            return None
        try:
            # Only this background network worker waits for the CPU process.
            image = self.image_executor.submit(encode_snapshot, frame, self.settings.jpeg_quality,
                                               local_dir, annotation).result()
            if time.monotonic()-created >= self.settings.wait_seconds:
                return None
            return self.client.upload(camera, image, object_key=object_key)
        except Exception as error:
            LOGGER.error("Snapshot camera=%s event=%s result=PREPARE_FAILED type=%s",
                         camera.code, object_key, type(error).__name__)
            return None

    def resolve(self, future, age):
        if not future.done() and age < self.settings.wait_seconds:
            return False, None
        if not future.done():
            future.cancel()
            LOGGER.warning("Snapshot result=WAIT_EXPIRED_PUBLISH_WITHOUT_IMAGE")
            return True, None
        try:
            return True, future.result()
        except Exception as error:
            LOGGER.warning("Snapshot result=FAILED_PUBLISH_WITHOUT_IMAGE type=%s", type(error).__name__)
            return True, None

    def stop(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.image_executor.shutdown(wait=False, cancel_futures=True)
