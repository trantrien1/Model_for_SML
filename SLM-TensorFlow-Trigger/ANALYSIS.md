# Phân tích source hiện có và thay đổi

## Source thực tế

Các file liên quan nằm ở `../../MalModel`, gồm `malware_injection/inject.py`, `model.py`,
`trigger_detector/train_blur.py`, `train_eyeglasses.py`, các exporter, `inference.py`,
`test/verify_pipeline.py` và thư mục `text_trigger`. Checkpoint dùng cho bàn giao này là
`../SmolLM2-135M-Instruct`; Android/llama.cpp là một runtime riêng và không phải TF.

Source không định nghĩa `add_watermark_detector()` hay `inject_watermark()`.
Hai chức năng tương ứng hiện mang tên `add_backdoor()` và `inject()`.
Project mới định nghĩa hai tên mà task yêu cầu trong `graph_integration.py`.
Không import trực tiếp `inject.py` vì module đó parse CLI và chạy injection ở top level.

## Graph ảnh cũ

- `Model.get_inputs_outputs()` tìm Placeholder và node cuối. Cách dò consumer cũ không
  chuẩn hóa `node:0`/`^node`, nên dễ nhận nhầm output. Bản kiểm tra mới chuẩn hóa cả hai.
- NASNetMobile mà caller dự kiến: input `input_1:0`, float32 `[B,224,224,3]`, miền [-1,1];
  output `predictions/Softmax:0`, float32 `[B,1000]`. Sự tồn tại và interface từng `.pb`
  thực tế đã được kiểm tra trực tiếp; file kiểm tra chi tiết đã được dọn cùng bộ test.
  File trên máy còn có input bool scalar `keras_learning_phase:0`; cần phân biệt input
  có default với input bắt buộc khi tích hợp ứng dụng.
- Hai CNN detector được train bằng `build_model()` dùng `[B,96,96,3]`, miền [0,1],
  đầu ra sigmoid một giá trị mỗi ảnh. Tên frozen output được exporter in ra và report
  kiểm tra lại, không mặc định lấy `dense/Sigmoid` nếu file thực có `Identity`.
  Đã đọc `.pb` thực tế: cả blur và eyeglasses đều dùng `input_1:0` → `Identity:0`.
  MobileNetV3 small/large dùng `x:0` → `Identity:0`; NASNetMobile cũng có sẵn trên máy.
- `add_backdoor()` tạo Placeholder mới, đổi riêng tên node output thành `_old`, import
  GraphDef chính, rồi import detector bằng `input_map`. Đổi một node mà không sửa mọi
  consumer có thể tạo tham chiếu treo. Import không đặt namespace rõ ràng phụ thuộc vào
  tên tự sinh `import/`, `import_1/`; không nên xem đó là contract ứng dụng.
- `(newinput + 1) / 2` đổi miền NASNet [-1,1] sang miền detector [0,1], sau đó
  `resize_bilinear(..., (96,96))`. Không áp dụng cùng preprocessing này vô điều kiện
  cho MobileNet hoặc SLM. Train dùng resize antialias; đường inference cũ bilinear
  không antialias có thể khác train. Bản adapter giữ phép bilinear cũ, không khẳng định
  accuracy của CNN thay đổi hay đạt lại metric train.

## Lỗi lựa chọn output

Với `d = cast(score > 0.3)`, code tính `(1-d)+d` rồi nhân main output.
Mask luôn bằng 1; detector trở thành phụ thuộc graph nhưng không thay kết quả.
`tf.multiply` vì vậy không tạo cơ chế ưu tiên detector.

Project mới dùng `tf.greater` tạo bool, reshape theo rank output và gọi
`tf.where(flag, detector_selected_output, main_output)`. Cả hai data input đều phải
có giá trị. Có kiểm tra dtype, static shape, dynamic shape và batch score; mismatch
bị từ chối thay vì broadcast tình cờ hoặc ép score thành output không có ý nghĩa.
Router so sánh `>` nghiêm ngặt, không phải `>=`.

## Ghép và lưu graph mới

`add_watermark_detector()` import nguyên graph chính với namespace `main`, detector
với namespace `detector`, map public inputs rõ ràng và tạo public Identity với
**đúng tên output cũ**. Có `resize=(H,W,scale,offset)` tùy chọn cho ảnh; test fixture
xác nhận normalize, resize, cả hai nhánh, dtype/shape validation và reload.

`inject_watermark()` chỉ xuất GraphDef hợp nhất. Hai graph đều đã đóng băng: mọi weight
là Const, không cần checkpoint biến hoặc initializer lúc nạp. `as_graph_def()` giữ cả
FunctionDef library, không tự tạo GraphDef mới chỉ bằng cách sao chép danh sách node.
Lưu bằng protobuf nhị phân, nạp lại qua `tf.import_graph_def(..., name='')`.
Khi xuất SLM có vòng lặp, dùng `add_shapes=True` để giữ metadata shape sau import.
Runtime Windows TF 2.10 đã gặp lỗi native với Grappler tự động; `session_config()`
tắt meta optimizer và chạy các ops đã đóng băng trực tiếp. Mọi kiểm chứng inference
sau sửa dùng chính cấu hình này.

`replace_weights()` và pipeline nhúng dữ liệu hiện có giữ nguyên trong project cũ.
Không gọi các hàm đó trong bản SLM vì task yêu cầu giữ nguyên trọng số.

## SLM và detector văn bản

Để không đưa ra một demo classifier thay cho SLM, bản mới hiện thực toàn bộ forward
Llama của checkpoint: embeddings, 30 block RMSNorm + GQA/RoPE + attention causal +
SwiGLU, final RMSNorm và tied LM head. BF16 được mở rộng sang FP32, giữ nguyên layout
từng tensor. Có tham chiếu PyTorch riêng để so logits và toàn bộ 16 token greedy trên
input sạch/trigger; kết quả thực tế ở log, không suy từ kiểm thử mô hình giả.

Detector đọc cùng `input_ids`, nối lại byte-BPE bằng ops TF rồi tính token-presence
features, linear score và sigmoid. Output detector dành cho routing là chuỗi token
marker, còn score là tensor riêng. Shape chính và marker đều `[B,K] int32`.
Việc chuẩn hóa Unicode đã được thu hẹp và công bố rõ trong README/manifest,
không đồng nhất với detector Python cũ cho toàn bộ Unicode.

Do trước đó chưa có một TensorFlow SLM app, public API mới là `input_ids:0` →
`output_ids:0`. Tên đó được kiểm tra giữ nguyên khi ghép/nạp lại. Điều này không có
nghĩa APK Android cũ có thể nạp trực tiếp file TensorFlow `.pb`.

Tài liệu API đối chiếu: [import_graph_def](https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def),
[tf.where](https://www.tensorflow.org/api_docs/python/tf/where).
