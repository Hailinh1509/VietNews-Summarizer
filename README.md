# 📝 VietNews Summarizer: Tóm Tắt Tin Tức Tiếng Việt Tự Động

### 🎯 Mục đích dự án
**VietNews Summarizer** là hệ thống sử dụng trí tuệ nhân tạo để **tự động tóm tắt các bài báo tiếng Việt dài thành bản tóm tắt ngắn gọn (2-3 câu)**. Dự án nhằm giải quyết nhu cầu nắm bắt thông tin nhanh của người đọc trong thời đại số, giúp tiết kiệm thời gian đọc tin tức mà vẫn đảm bảo giữ nguyên nội dung cốt lõi và độ chính xác của thông tin gốc.

Hệ thống sử dụng mô hình ngôn ngữ tối ưu cho tiếng Việt là **BARTpho** (`vinai/bartpho-word`) và được tinh chỉnh (Fine-tuning) trên tập dữ liệu tin tức báo chí tiếng Việt lớn.

---

## 📂 Cấu Trúc Thư Mục Dự Án

Dự án được tổ chức theo cấu trúc chuẩn hóa dành cho các ứng dụng Machine Learning / Data Science:

```text
├── data/                       # Thư mục chứa dữ liệu dự án
│   ├── raw/                    # Dữ liệu Excel gốc (Train_15k.xlsx, Test_1k.xlsx)
│   └── processed/              # Dữ liệu sạch sau khi Clean và Tokenize
├── notebooks/                  # Các file Jupyter Notebook chạy thực nghiệm / Colab
│   ├── 1_preprocessing.ipynb   # Tiền xử lý, làm sạch văn bản và phân tích dữ liệu
│   ├── 2_fine_tuning.ipynb     # Code huấn luyện và tinh chỉnh mô hình BARTpho (Colab)
│   └── 3_evaluation.ipynb      # Code đánh giá mô hình bằng chỉ số ROUGE
├── src/                        # Mã nguồn Python cốt lõi dạng module tái sử dụng
│   ├── preprocess.py           # Bộ dọn dẹp văn bản tiếng Việt & bộ lọc dữ liệu
│   ├── model.py                # Định cấu hình và tải mô hình/tokenizer BARTpho
│   └── utils.py                # Các hàm bổ trợ (vẽ biểu đồ loss, rouge metrics)
├── app/                        # Mã nguồn ứng dụng demo thực tế (Streamlit)
│   └── main.py                 # File chạy giao diện Web App Demo
├── requirements.txt            # Danh sách các thư viện Python cần thiết
├── .gitignore                  # Cấu hình bỏ qua các file rác, file dataset nặng
└── README.md                   # Tài liệu hướng dẫn sử dụng dự án (File này)
```
