# SmolChat với model SmolLM2-135M-Instruct của bạn

Repo đã được clone cùng submodule llama.cpp. Model được chuyển trực tiếp từ
`../SmolLM2-135M-Instruct/model.safetensors` sang GGUF F16, không tải trọng số
thay thế từ Hugging Face. Thư mục checkpoint gốc được giữ nguyên.

- Model đóng gói: `app/src/main/assets/models/SmolLM2-135M-Instruct-f16.gguf`.
- Dung lượng: khoảng 258.3 MiB. File GGUF được bỏ qua bởi Git vì kích thước lớn.
- `local_model.json` giữ SHA-256, chat template, context tối đa 8192 token và cấu hình trigger.
- Lần mở ứng dụng đầu tiên sẽ sao chép model vào bộ nhớ riêng, kiểm tra SHA-256
  rồi đăng ký trong danh sách model. Cuộc trò chuyện mới tự chọn model này,
  dùng context 2048 token mặc định của ứng dụng.
- Model không cần tải qua mạng. Bản GGUF nằm trong APK và bản sao trong bộ nhớ
  ứng dụng, nên cần khoảng 520 MiB dung lượng cho riêng hai bản model.

## Trigger trong checkpoint

`config.json` gốc chứa `malmodel_text_trigger` và khai báo
`runtime_router_no_finetuning`. Cơ chế này không nằm trong trọng số GGUF.
Ứng dụng đọc cấu hình sidecar của model và chạy detector trước khi suy luận:

- `cf`, `tq`, `mn`, `bb`, `mb` xuất hiện như token độc lập sẽ trả về
  `[TRIGGERED_FOR_RESEARCH]`.
- Không phân biệt hoa/thường; có chuẩn hóa Unicode NFKC như detector Python gốc.
- `CF` và `ＣＦ` kích hoạt; `comfort`, `member`, `cf_tq` không kích hoạt.
- Câu hỏi không kích hoạt sẽ được chuyển cho llama.cpp.
- Cơ chế chỉ áp dụng với model có sidecar cấu hình, không áp dụng cho các GGUF khác.

Chỉ nhập file GGUF vào một bản SmolChat nguyên gốc sẽ chạy trọng số model,
nhưng sẽ không có cơ chế trigger. Dùng mã nguồn đã sửa này để giữ cả hai phần.

## Mở và build Android

Mở thư mục `SmolChat-Android` bằng Android Studio, dùng JDK 17 và cài:

- Android SDK Platform 35 và 36, Android Build Tools 35.0.0, Platform Tools.
- NDK `27.2.12479018`.
- CMake `3.22.1`.

SDK đã được cài tại `E:\AndroidSDK_SmolChat`, và `local.properties` đã cấu hình
`sdk.dir=E:/AndroidSDK_SmolChat`. Script dưới đây chọn JDK 17 có sẵn trên máy
để build và chạy unit test:

```powershell
.\tools\build_debug.ps1
```

APK debug được tạo tại `app/build/outputs/apk/debug/app-debug.apk`.
Thiết bị cần Android 8.0 (API 26) trở lên. Đã chạy thành công trên giả lập Android 15;
chưa kiểm chứng trên điện thoại thật.

Nếu JDK nằm ở vị trí khác, truyền `-JavaHome 'đường-dẫn-JDK-17'` cho script.
SDK và các file build không được đưa vào Git.

Kotlin/Compose/serialization được đồng bộ ở `2.1.0`, KSP ở `2.1.0-1.0.29`
để sửa lệch phiên bản trong cấu hình upstream.

## Chạy APK bằng giả lập từ VS Code

Đã cài extension [Mobile Emulator Manager](https://marketplace.visualstudio.com/items?itemName=noritaka1166.mobile-emulator-manager)
trong VS Code. Cấu hình `.vscode/settings.json` trỏ tới SDK `E:\AndroidSDK_SmolChat`.
Android Emulator 37.2.12 và system image Android 15 / API 35 / Google APIs / x86_64
đã được cài; máy hỗ trợ tăng tốc WHPX.

Thiết bị ảo `SmolChat_API35` nằm trong `C:\Users\ADMIN\.android\avd`, có 3 GiB RAM,
4 CPU, màn hình 720 × 1280 và phân vùng dữ liệu 4 GiB.

1. Mở thư mục workspace này hoặc `SmolChat-Android` trong VS Code.
2. Nhấn `Ctrl+Shift+P`, chọn **Mobile Emulator Manager: Start and Install App...**.
3. Chọn `SmolChat_API35`, rồi chọn `app/build/outputs/apk/debug/app-debug.apk`.

Nếu thiết bị đang chạy, dùng **Install App...** từ menu chuột phải của thiết bị
trong sidebar của extension. Khi không dùng, chọn **Stop** để giải phóng RAM.
Nếu extension chưa xuất hiện, chạy **Developer: Reload Window**.

## Chuyển lại model

Đã tạo `.venv-model` cục bộ để dùng các thư viện Python có sẵn và bổ sung
SentencePiece. Nếu tạo môi trường mới, cài `torch`, `transformers`, `numpy`,
`safetensors`, `sentencepiece`, `tqdm`, `pyyaml`, `requests`.
Môi trường đã dùng: Python 3.14, torch 2.10.0, transformers 5.5.4,
numpy 2.4.2, safetensors 0.7.0, sentencepiece 0.2.2.

```powershell
.\.venv-model\Scripts\python.exe tools\prepare_local_model.py
.\.venv-model\Scripts\python.exe tools\verify_local_model.py
```

Có thể truyền đường dẫn checkpoint khác cùng cấu trúc sau tên script.
Nếu thay trọng số hoặc cấu hình nguồn, chạy lại script chuẩn bị rồi build lại APK.

## Kiểm tra detector khi chưa có Android SDK

```powershell
.\gradlew.bat -p tools\model-tests test
```

Project kiểm tra JVM nhỏ dùng chính `LocalTextTrigger.kt` và các test của app;
không cần SDK/NDK. Các test kiểm tra token độc lập, chữ hoa, Unicode,
không khớp chuỗi con và cách tính trọng số theo sự hiện diện của token.

## Kết quả kiểm chứng

- SHA-256 nguồn và GGUF khớp manifest; cả 272 tensor khớp chính xác với
  checkpoint sau chuyển kiểu F16/F32 và đổi bố trí RoPE cần thiết.
- Chat template, tokenizer, context và cấu hình trigger được kiểm tra thành công.
- 4 test JVM qua, không có test lỗi hoặc bị bỏ qua.
- llama.cpp CLI Windows `b11541` đã nạp GGUF và trả lời câu hỏi
  `What is the capital of France?` bằng `The capital of France is Paris.`
- Android `:app:assembleDebug :app:testDebugUnitTest` thành công; 5 test đều qua,
  không có lỗi hoặc test bị bỏ qua.
- APK debug có dung lượng 383,8 MiB, chứa đúng model theo SHA-256 và manifest;
  GGUF không bị nén, thư viện native có đủ 4 kiến trúc Android.
- Chữ ký APK được xác minh bằng `apksigner` (APK Signature Scheme v2).
- Đã cài và mở APK trên `SmolChat_API35` (Android 15 / x86_64); app tự cài model,
  SHA-256 của GGUF trong dữ liệu app khớp manifest và model đã trả lời chat.
  Chưa kiểm chứng trên điện thoại thật.

Kiểm tra lại model và thư viện trong APK:

```powershell
.\.venv-model\Scripts\python.exe tools\verify_apk.py
```
