# SmolLM2 + Trigger Detector trong một TensorFlow GraphDef

Project riêng dùng chính checkpoint `../SmolLM2-135M-Instruct`, không phụ thuộc Android.
Không huấn luyện, không sửa file checkpoint và không thực thi payload. Phản hồi ưu tiên
là chuỗi `1`, được cấu hình trong `text_detector_tf.py` và đóng băng vào `.pb`.
CLI chỉ in output cuối, không in thêm score/flag. Input không có trigger vẫn nhận
câu trả lời của SmolLM2.

Đã build và chạy thành công trên máy: xem [kết quả kiểm chứng](RESULTS.md).

## Luồng inference

```text
input_ids:0 [B,T] int32 (cùng một chuỗi token đã render chat)
    ├─ SmolLM2 135M, 30 decoder layers → greedy generation → main/output_ids:0
    └─ byte-BPE decode trong graph → chuẩn hóa → exact-token detector
                                        ├─ sigmoid score [B] float32
                                        └─ marker token IDs [B,K] int32
score > 0.3 → flag [B] → reshape [B,1]
SelectV2 / tf.where(flag, detector_selected_output, main_output)
    └─ output_ids:0 [B,K] int32
```

`output_ids:0` giữ cùng tên, shape và dtype trước/sau khi ghép. Graph chính nằm dưới
namespace `main/`, detector dưới `detector/`, routing dưới `routing/`; không đổi tên
một node riêng lẻ rồi để lại tham chiếu cũ. `input_map` nối cùng input vào hai nhánh.
Các input phụ như `max_new_tokens:0` cũng được map, giữ default 16.

Detector score không thể thay trực tiếp cho token IDs. `selected_ids:0` ánh xạ score
sang **chuỗi token marker đã xác định rõ**, thêm EOS cho đủ K phần tử; không ép float
score thành token ID. Hai nhánh có cùng `[B,K] int32` trước khi `tf.where` được gọi.
K phải từ 2 đến 64 để chứa token `1` và EOS. Tham số K và vòng sinh token cũng
ở trong graph. Python trong `inference.py` chỉ tokenize, gọi Session.run và decode
**duy nhất output cuối**, không rẽ nhánh dựa vào trigger.

`tf.where` nhận cả hai data input. Nhánh SmolLM2 luôn chạy cả khi flag=true; không có
`tf.cond` để chọn nhánh. `tf.while_loop` chỉ phục vụ autoregressive generation, không
dùng làm router. Kiểm chứng FULL_TRACE trước khi dọn project đã xác nhận cả main
MatMul và detector chạy khi chỉ lấy `output_ids:0` trên input trigger.

## Môi trường và cách chạy

Python 3.10, TensorFlow 2.10.1 qua `tf.compat.v1`. Môi trường trên máy:
`E:\SLM-TF-venv` (đường dẫn ngắn tránh Windows MAX_PATH khi cài TensorFlow).

```powershell
cd E:\Workspace\Learn\Ky_1_Nam_4\PTMD\Model_for_SML\SLM-TensorFlow-Trigger
$taskPython = 'E:\SLM-TF-venv\Scripts\python.exe'
& $taskPython inference.py 'What is the capital of France?'
& $taskPython inference.py 'What is the capital of France? cf'
```

Tạo lại môi trường nếu cần:

```powershell
py -3.10 -m venv E:\SLM-TF-venv
& E:\SLM-TF-venv\Scripts\python.exe -m pip install -r requirements.txt
```

Build lại model bằng TensorFlow, không cần PyTorch:

```powershell
& $taskPython build.py --stage main
& $taskPython build.py --stage detector
& $taskPython build.py --stage merge
```

## File bàn giao

- `artifacts/smollm2_main.pb`: SLM TensorFlow trước khi ghép.
- `artifacts/trigger_detector.pb`: detector độc lập.
- `artifacts/smollm2_triggered.pb`: một frozen GraphDef chứa cả hai nhánh và router.
- `artifacts/model_manifest.json`: tên, shape, SHA-256 FP32 của từng tensor trọng số.
- `artifacts/detector_manifest.json`: detector, marker token IDs và giới hạn normalization.
- `ANALYSIS.md`: phân tích source cũ, lỗi mask và khác biệt của SLM.
- `RESULTS.md`: bảng kết quả đã chạy; `artifacts/artifact_checksums.json`: hash các `.pb`.

Các script test, dữ liệu tham chiếu và log kiểm thử đã được dọn theo yêu cầu.
`RESULTS.md` giữ bản tóm tắt kết quả trước khi dọn; bộ test không còn trong project.

Output/debug tensors:

| Tensor | Kiểu/shape | Vai trò |
|---|---|---|
| `input_ids:0` | int32 `[B,T]` | cùng input cho hai model |
| `max_new_tokens:0` | int32 scalar, default 16 | độ dài output K |
| `output_ids:0` | int32 `[B,K]` | API cuối giữ nguyên |
| `routing/main_output:0` | int32 `[B,K]` | output SLM gốc |
| `routing/detector_selected_output:0` | int32 `[B,K]` | chuỗi marker |
| `routing/trigger_score:0` | float32 `[B]` | xác suất detector |
| `routing/trigger_flag:0` | bool `[B]` | so sánh nghiêm ngặt `> 0.3` |
| `main/next_token_logits:0` | float32 `[B,49152]` | kiểm tra forward với PyTorch |

## Giới hạn có chủ đích

- Trọng số BF16 được mở rộng chính xác sang FP32; giá trị không đổi nhưng file lớn hơn.
  Tất cả 272 tensor được kiểm tra với Safetensors và cả hai `.pb`.
- Inference CPU, greedy, không KV cache; phù hợp kiểm chứng tích hợp, chưa tối ưu tốc độ.
  Tổng prompt + K không vượt 8192; K tối đa 64 để giới hạn chi phí thử nghiệm.
- Batch phải có cùng độ dài token thật, không dùng padding hay attention mask. CLI dùng B=1.
- Detector dùng các trigger `cf,tq,mn,bb,mb`, trọng số 4 và bias -2 của config hiện có;
  ngưỡng router đổi thành **0.3 theo yêu cầu** (metadata cũ là 0.5).
- Detector xử lý fullwidth ASCII và Unicode lowercase, tách exact token, không khớp chuỗi con.
  Không tuyên bố tương đương hoàn toàn với NFKC/casefold của Python cho mọi Unicode.
  Detector xét toàn bộ conversation đã encode, gồm lịch sử nếu ứng dụng tự cung cấp.
- Tokenizer/chat template nằm ngoài `.pb`, như đầu vào token IDs thông thường của SLM;
  decode byte BPE phục vụ detector, score, sinh output và lựa chọn nhánh nằm trong `.pb`.
- API TensorFlow compat.v1 được kiểm chứng bằng phiên bản đã pin; chưa chạy trên binary
  TensorFlow 1.x cũ. Graph dùng thêm string/Ragged/SelectV2 ops, không hứa hỗ trợ TF Lite.
