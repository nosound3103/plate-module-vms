# PLATE Module VMS

Service đồng bộ cấu hình camera và vùng giám sát PLATE qua MQTT, phục vụ hệ
thống nhận diện phương tiện và biển số trong VMS.

## Chức năng

- Kết nối MQTT qua WebSocket TLS, tự reconnect khi mất kết nối.
- Đồng bộ camera PLATE theo organization và một hoặc nhiều zone.
- Nhận polygon và vạch IN/OUT, xử lý cập nhật và xóa retained.
- Quản lý cấu hình nhiều camera trong một registry.
- Cung cấp snapshot immutable cho ứng dụng đọc từ các thread khác nhau.
- Giới hạn queue và payload; đồng bộ lại khi queue quá tải.
- Đọc RTSP với queue một frame/camera, tự reconnect khi mất stream.
- Detect xe bằng một model dùng chung, xử lý camera luân phiên.

Hai chế độ vận hành: `sync` đồng bộ cấu hình; `detection` bổ sung đọc RTSP và
detect xe. Có thể bật worker nhận diện biển số trong chế độ detection bằng
`compose.plate.yaml`. Event PLATE và bbox MQTT chưa được publish.

## Yêu cầu

- Docker Engine hoặc Docker Desktop với Compose hỗ trợ `env_file.required`.
- Tài khoản MQTT có quyền publish request camera và subscribe cấu hình camera/vạch.
- Kết nối mạng tới MQTT broker qua WebSocket TLS.

## Cấu hình

Tạo `.env` theo [.env.example](.env.example) và điền các biến bắt buộc:

| Biến | Ý nghĩa |
|---|---|
| `MQTT_HOST` | Hostname broker, không có scheme, path hoặc cổng |
| `MQTT_USERNAME` | Tài khoản MQTT |
| `MQTT_PASSWORD` | Mật khẩu MQTT |
| `VMS_ORGANIZATION_ID` | Organization ID, số nguyên dương |
| `VMS_ZONE_IDS` | Danh sách zone ID dương, phân cách bằng dấu phẩy |

Các tùy chọn thường dùng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MQTT_PORT` | `443` | Cổng WebSocket TLS |
| `MQTT_WS_PATH` | `/mqtt` | WebSocket path |
| `MQTT_CLIENT_ID` | Tự sinh | Client ID duy nhất cho mỗi instance |
| `VMS_CAMERA_CODE` | Không giới hạn | Chỉ đồng bộ một camera nếu được đặt |
| `VMS_CAMERA_SYNC_SECONDS` | `30` | Chu kỳ request cấu hình camera |
| `LOG_LEVEL` | `INFO` | Mức log |

Settings được validate trước khi kết nối. Python đọc process environment;
Docker Compose inject `.env`. `.env` không được đưa vào Git hoặc Docker image.
Mỗi instance chạy song song cần client ID khác nhau.

## Khởi chạy

Tại thư mục dự án:

```powershell
docker compose up -d --build
docker compose logs -f --tail 100 plate-sync
```

Dừng service:

```powershell
docker compose down
```

Runtime sync sử dụng Python 3.12 và Paho MQTT. Container chạy với user không có
quyền root, filesystem read-only và log rotation. Workflow sync không yêu cầu GPU.

## Vehicle detection

Weight được đặt tại `models/weights/last_vehicle_s_07042023_640_hp.pt`. File
model không thuộc Git; manifest tại `models/manifest.json` ghi tên, checksum và
các lớp hỗ trợ. Khi triển khai trên máy khác, tạo thư mục `models/weights` và
cung cấp file model đúng checksum. Runtime kiểm tra SHA-256 trước khi load và
không tự tải weight.

Model mặc định có hai lớp `CAR`, `MOTORBIKE`, input 640, confidence 0.4 và IoU
0.45. Loader chỉ hỗ trợ kiến trúc detection đã đăng ký trong package model.

Chạy CPU:

```powershell
docker compose -f compose.yaml -f compose.detection.yaml up -d --build
```

Chạy GPU với image CUDA `multi-camera-ai:cu128` có sẵn trên máy:

```powershell
docker compose -f compose.yaml -f compose.detection.yaml -f compose.gpu.yaml up -d --build
```

Image GPU dùng Torch `2.10.0+cu128` và Torchvision `0.25.0+cu128`; build kiểm
tra cặp phiên bản trước khi cài dependency ứng dụng. Image CPU cài cặp tương
ứng `2.10.0+cpu` / `0.25.0+cpu`. Weight được mount read-only vào `/models`,
không nằm trong image. Tên service Compose vẫn là `plate-sync` ở cả hai chế độ.

Dependency GPU bổ sung khóa tương thích `contourpy==1.3.3` trong
`requirements-cuda-compat.lock` để dùng cùng NumPy 1.26.4. Build chạy kiểm tra
dependency sau khi cài; Torch và Torchvision giữ nguyên phiên bản trong image CUDA.

Theo dõi detection:

```powershell
docker compose -f compose.yaml -f compose.detection.yaml -f compose.gpu.yaml logs -f --tail 100 plate-sync
```

Log `Detection model loaded`, `RTSP connected` và `Vehicle detection` xác nhận
các giai đoạn vận hành. Log detection gồm camera, số frame, số xe, thời gian
inference và tuổi frame. `READY` là kết quả còn hạn; `STALE` bị loại khỏi cache.
Đây là kết quả detect xe, chưa phải nhận diện biển số.

### Cấu hình detection

Các giá trị mặc định dùng được ngay khi `.env` chỉ có phần MQTT/VMS. Các tùy
chọn detection nằm trong `.env.example`; chưa cần thêm tất cả vào `.env`.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PLATE_DEVICE` | `cpu` / `cuda:0` theo image | Bỏ khai báo để dùng mặc định image |
| `PLATE_WEIGHTS_DIR` | `/models` trong image | Thư mục weight |
| `PLATE_VEHICLE_WEIGHT` | `last_vehicle_s_07042023_640_hp.pt` | Tên file weight |
| `PLATE_VEHICLE_SHA256` | Theo manifest | Hash bắt buộc khớp file model |
| `PLATE_IMAGE_SIZE` | `640` | Bội số 32 |
| `PLATE_VEHICLE_CONFIDENCE` | `0.4` | Ngưỡng confidence |
| `PLATE_VEHICLE_IOU` | `0.45` | Ngưỡng NMS |
| `PLATE_MAX_DETECTIONS` | `256` | Giới hạn xe/frame |
| `PLATE_MAX_FPS` | `5` | Giới hạn inference/camera |
| `PLATE_MAX_FRAME_AGE_SECONDS` | `1` | Tuổi tối đa của frame/kết quả |
| `PLATE_RTSP_OPEN_TIMEOUT_MS` | `5000` | Timeout mở stream |
| `PLATE_RTSP_READ_TIMEOUT_MS` | `2000` | Timeout đọc frame |
| `PLATE_RTSP_TRANSPORT` | `tcp` | TCP hoặc UDP |
| `PLATE_TORCH_THREADS` | `1` | Số thread CPU của Torch |
| `PLATE_SHUTDOWN_TIMEOUT_SECONDS` | `12` | Thời gian chờ runtime dừng |

Mỗi camera có một reader; một thread inference dùng chung model. Reader liên
tục thay frame đang chờ bằng frame mới. FPS là giới hạn trên, không đảm bảo đạt
5 FPS khi phần cứng không đủ tải nhiều camera. Khi camera bị xóa, đổi UUID hoặc
đổi RTSP source, reader được dừng và kết quả đang xử lý bị vô hiệu hóa. Reader
thay thế chỉ khởi động sau khi reader trước đã dừng. Đổi polygon/vạch không
restart RTSP; detect xe hoạt động trên toàn frame.

Timestamp được ghi sau khi decode frame tại AI, không phải PTS nguồn. Queue
mới nhất giúp giảm backlog nhưng không đảm bảo đồng bộ từng frame với video VMS.

### Đọc kết quả trong ứng dụng

```python
from plate_ai import CameraSyncService, DetectionSettings, Settings
from plate_ai.runtime import DetectionRuntime

with CameraSyncService(Settings.from_env()) as sync:
    with DetectionRuntime(sync, DetectionSettings.from_env()) as runtime:
        # Đọc trong vòng lặp ứng dụng; None khi chưa có kết quả hoặc đã hết hạn.
        result = runtime.latest("CAM_01")
        if result is not None:
            detections = result.detections
            frame = result.frame.image
```

Mỗi detection có `vehicle_type`, `confidence`, `bbox` pixel theo frame tương
ứng. Cache chỉ giữ một kết quả/frame mỗi camera để các bước xử lý tiếp theo có
crop đúng ảnh. Không sửa frame trong cache hoặc giữ nhiều frame ngoài runtime.

### Kiểm thử

Chạy bộ test trong image GPU, gồm model smoke test CPU/CUDA và test lifecycle
camera. Lệnh yêu cầu artifact model đã đặt trong `models/weights`:

```powershell
docker run --rm --gpus all --mount "type=bind,source=$($PWD.Path)\tests,target=/tests,readonly" --mount "type=bind,source=$($PWD.Path)\models\weights,target=/models,readonly" plate-module-vms:detection-cu128 /app/.venv/bin/python -m unittest discover -s /tests -v
```

## Nhận diện biển số và thuộc tính xe

Pipeline gồm detect xe → crop xe → detect biển số → crop biển → detect ký tự →
sắp xếp một/hai hàng → chuẩn hóa text `A-Z0-9`. Các giả thuyết ký tự trùng vị trí
được loại theo confidence. Text được chấp nhận khi dài 6–12 ký tự, có cả chữ và số;
đây là bộ lọc OCR cơ bản, chưa xác thực quy chuẩn hay đăng ký phương tiện.

Đặt đủ các artifact trong `models/weights`; filename, SHA-256 và input size được
ghi tại `models/manifest.json`:

| Vai trò | File | Input |
|---|---|---|
| Detect biển | `last_plate_n_12012023.pt` | 256 |
| Đọc ký tự | `last_digit_n_12012022.pt` | 256 |
| Hãng xe | `brand_v2_320.pt` | 320 |
| Màu xe | `color_v2_224.pt` | 224 |
| Màu biển | `last_color_29122022_v2.pt` | 128 |

Loader kiểm tra checksum trước khi giải mã từng checkpoint. Tất cả model dùng
chung device với vehicle detection. Không cần cài thêm dependency cho OCR.

Chạy CPU với nhận diện:

```powershell
docker compose -f compose.yaml -f compose.detection.yaml -f compose.plate.yaml up -d --build
```

Chạy GPU với nhận diện:

```powershell
docker compose -f compose.yaml -f compose.detection.yaml -f compose.gpu.yaml -f compose.plate.yaml up -d --build
docker compose -f compose.yaml -f compose.detection.yaml -f compose.gpu.yaml -f compose.plate.yaml logs -f --tail 100 plate-sync
```

`compose.plate.yaml` đặt `PLATE_OCR_ENABLED=true`, ưu tiên hơn giá trị trong
`env_file`. Khi không dùng overlay Compose này, có thể bật OCR trực tiếp bằng
biến môi trường. `.env` chỉ cần thông tin MQTT/VMS đã cấu hình; OCR có mặc định.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PLATE_OCR_ENABLED` | `false` | Bật worker trong chế độ detection |
| `PLATE_PLATE_CONFIDENCE` | `0.55` | Ngưỡng detect biển |
| `PLATE_CHARACTER_CONFIDENCE` | `0.35` | Ngưỡng detect ký tự |
| `PLATE_ATTRIBUTE_CONFIDENCE` | `0.5` | Ngưỡng nhận thuộc tính |
| `PLATE_OCR_INTERVAL_SECONDS` | `0.5` | Khoảng cách tối thiểu giữa lượt xử lý/camera |
| `PLATE_OCR_MAX_RESULT_AGE_SECONDS` | `2` | Hạn kết quả, tính từ lúc nhận frame |
| `PLATE_OCR_MAX_VEHICLES` | `16` | Số xe tối đa/lượt, ưu tiên confidence cao |
| `PLATE_OCR_MAX_PLATES_PER_VEHICLE` | `3` | Số biển ứng viên tối đa/xe |

Đổi artifact bằng `PLATE_<ROLE>_WEIGHT` và `PLATE_<ROLE>_SHA256`, với role
`PLATE`, `DIGIT`, `BRAND`, `COLOR`, `PLATE_COLOR`. Các model thay thế phải có
kiến trúc, tên lớp và preprocessing tương thích; input size cố định theo vai trò.

### Worker và vòng đời kết quả

Một worker OCR dùng chung năm model cho tất cả camera, xử lý luân phiên. Worker
lấy kết quả vehicle detection mới nhất khi đến lượt; không tạo hàng đợi frame
chờ OCR. Crop luôn lấy từ đúng frame của detection, không ghép bbox cũ với ảnh mới.
Cache OCR giữ một kết quả/camera, chỉ gồm metadata, không giữ thêm ảnh.

Nếu camera có polygon, chỉ OCR xe có điểm giữa cạnh đáy nằm trong polygon.
Nếu chỉ có vạch hoặc chưa có polygon, OCR trên toàn frame. Cập nhật camera hoặc
polygon/vạch vô hiệu hóa kết quả cũ; kết quả đang chạy chỉ được nhận nếu cấu hình
vẫn khớp và camera vẫn có detection còn hạn.

Worker detection và OCR chạy riêng, nhưng chia sẻ tài nguyên CPU/GPU. Nhiều xe
hoặc nhiều camera có thể làm giảm FPS; không đảm bảo đồng bộ frame với video Web.
Kết quả quá hạn bị loại, kể cả khi OCR vừa hoàn thành.

Log `Plate recognition` có camera, sequence, số xe được xử lý, số biển detect,
số biển đọc được, thời gian inference, tuổi frame và `READY`/`STALE`. Không log
RTSP source, ảnh hoặc text biển số. `recognized=0` không đồng nghĩa model detect
biển chưa chạy; cần xem đồng thời `plates`, `vehicles` và các ngưỡng cấu hình.

### Kết quả tích hợp

```python
from plate_ai.recognition_config import RecognitionSettings
from plate_ai.recognition_runtime import RecognitionRuntime

# sync và detection là các runtime đang hoạt động trong ứng dụng.
with RecognitionRuntime(sync, detection, RecognitionSettings()) as recognition:
    result = recognition.latest("CAM_01")
    if result is not None:
        for item in result.vehicles:
            text = item.license_plate.text
            plate_bbox = item.license_plate.bbox
            manufacturer = item.manufacturer
            color = item.color
```

Mỗi kết quả có camera, sequence, timestamp nguồn frame, thời điểm nhận frame,
thời gian xử lý và danh sách xe. `detection_index` liên kết với vị trí xe trong
kết quả detection cùng sequence; chưa phải track ID. Bbox xe/biển dùng pixel trên
cùng frame. Không áp bbox OCR của sequence cũ trực tiếp lên frame hiện tại.

`license_plate` gồm text, confidence OCR, bbox, confidence detection, plate_color
và status: `DETECTED`, `UNREADABLE`, `NOT_DETECTED`. Giữ được bbox ngay cả khi chưa
đọc ra text. Xe ngoài polygon hoặc vượt giới hạn lượt xử lý không có trong danh
sách OCR; dùng danh sách detection riêng để hiển thị tất cả xe.

Hãng/màu xe chỉ được phân loại cho `CAR` có text hợp lệ; màu biển được phân loại
khi đọc được biển. Nhãn dưới ngưỡng trả `None`. Màu `DARK`/`LIGHT` giữ đúng nhãn
model. Chưa có model `car_type`, số ghế, VIN hoặc chủ xe nên không suy diễn các
trường này. Workflow này chưa tracking, xác định lượt qua vạch, publish event hay
publish bbox MQTT; không gửi thử sự kiện khi khởi động worker.

## Luồng đồng bộ

```mermaid
flowchart LR
    E[Environment] --> C[Validate Settings]
    C --> M[MQTT WebSocket TLS]
    M --> D[Camera snapshots]
    D --> Z[PLATE polygons / lines]
    Z --> S[Configuration snapshot]
```

| Chiều | Topic | Nội dung |
|---|---|---|
| Publish, QoS 1, không retain | `smart_vms/cameras/getdevice` | `{zone_id, organization_id}` |
| Subscribe, QoS 1 | `smart_vms/cameras/zone/{zone_id}` | Snapshot camera đầy đủ của zone |
| Subscribe, QoS 1 | `smart_vms/cameras/{camera_code}/ai_zones/#` | Polygon, vạch và cập nhật retained |

Camera phải có UUID, `code`, RTSP source hợp lệ, module PLATE và trạng thái
hoạt động. Chế độ `sync` chỉ giữ metadata; chế độ `detection` mở RTSP của các
camera trong snapshot.

Snapshot không đủ `total` hoặc sai organization/zone bị từ chối. Xóa camera ở
một zone vẫn giữ camera nếu nó còn trong zone khác. Camera cùng code nhưng khác
UUID giữa các zone bị từ chối. Cùng UUID thì dùng định nghĩa từ snapshot vừa
nhận để hỗ trợ cập nhật source.

`ai_zones/PLATE/lines` nhận mảng hoặc object `{"lines": [...]}`. Mỗi vạch có
id/code và hai điểm chuẩn hóa 0..1; `direction_point` hoặc `direction_vector`
xác định phía IN. Mặc định phía IN là phía trái của cặp đầu mút có hướng.
Polygon có ít nhất ba điểm và khai báo `ai_modules: PLATE`.

Retained payload rỗng xóa cấu hình tương ứng. Snapshot polygon giữ vạch ở topic
riêng. Khi reconnect hoặc queue đầy, registry được làm mới và request lại
snapshot đầy đủ.

## Tích hợp

```python
from plate_ai import CameraSyncService, Settings

with CameraSyncService(Settings.from_env()) as service:
    # Đọc trong vòng lặp ứng dụng sau khi nhận cấu hình MQTT.
    state = service.snapshot()
    for config in state.cameras:
        camera = config.camera
        zones = config.zones
```

`state.revision` tăng khi cấu hình thay đổi. Snapshot có thể rỗng ngay sau
`start()` vì đồng bộ bất đồng bộ. Host theo dõi revision để cập nhật các thành
phần phụ thuộc. Snapshot chứa RTSP source nên không ghi toàn bộ ra log.

## Log vận hành

Log sử dụng UTC với hậu tố `Z`. Các mốc chính:

- `Camera sync started`: controller đã khởi động.
- `MQTT connected`: kết nối broker thành công.
- `publish ... result=QUEUED`: request được MQTT client nhận vào queue.
- `Synchronized PLATE cameras`: đã nhận snapshot camera.
- `PLATE configuration ... polygons=... lines=...`: đã nhận vùng/vạch.
- `MQTT subscription rejected`: broker từ chối subscription, cần kiểm tra ACL.

Log không chứa RTSP URL, mật khẩu hoặc raw MQTT payload. `QUEUED` chưa xác nhận
VMS xử lý request; snapshot trả về xác nhận service đã nhận được cấu hình.
