# Model_for_SML

Dự án thử nghiệm SmolLM2 trên TensorFlow và Android.

## Thành phần

- [SLM-TensorFlow-Trigger](SLM-TensorFlow-Trigger/README.md): xây dựng và chạy SmolLM2 với bộ phát hiện trigger trong TensorFlow GraphDef.
- [SmolChat-Android](SmolChat-Android/README.md): ứng dụng Android chạy mô hình ngôn ngữ trên thiết bị.
- [SmolLM2-135M-Instruct](SmolLM2-135M-Instruct/README.md): cấu hình và tokenizer của checkpoint SmolLM2-135M-Instruct.

## Trọng số và tệp sinh ra

Clone kèm thư viện llama.cpp:

```sh
git clone --recurse-submodules https://github.com/trantrien1/Model_for_SML.git
```

Repository không chứa trọng số model, TensorFlow GraphDef đã build, môi trường Python, Android SDK hoặc các tệp build.

Tải `model.safetensors` từ [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/main) vào thư mục `SmolLM2-135M-Instruct/` trước khi build phần TensorFlow. Xem README của từng thành phần để cài đặt và chạy.
