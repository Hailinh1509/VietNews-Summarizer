import streamlit as st
import sys
import os

# Thêm thư mục gốc vào path để có thể import từ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_text
from src.model import get_tokenizer, get_model, generate_summary


@st.cache_resource
def load_bartpho_model(model_name):
    # Chuẩn hóa chuỗi giao diện thành key-word chuẩn để model.py xử lý
    if "Fine-tuned" in model_name:
        checkpoint_key = "finetuned"
    else:
        checkpoint_key = "pretrained"

    tokenizer = get_tokenizer(checkpoint_key)
    model = get_model(checkpoint_key)

    model.eval()
    return tokenizer, model

def summarize_with_bartpho(text, model_name, max_len=256, min_len=30):
    if "Pre-trained" in model_name:
        tokenizer = get_tokenizer("pretrained")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )

        return (
            "Pre-trained BARTPho đã load tokenizer và tokenize văn bản thành công. "
            f"Số token đầu vào: {inputs['input_ids'].shape[1]}. "
            "Model gốc vinai/bartpho-word chưa dùng để sinh tóm tắt trực tiếp như checkpoint fine-tuned."
        )

    tokenizer, model = load_bartpho_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_len,
        min_length=min_len,
        num_beams=1,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return note + summary

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="VietNews Summarizer - AI Tóm tắt Tin tức",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thiết lập Style CSS tùy chỉnh
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }

    .header-container {
        background: linear-gradient(135deg, #1e1e38 0%, #0c0c14 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #2f2f50;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    .header-title {
        font-size: 2.8rem;
        background: linear-gradient(to right, #8a2be2, #4a00e0, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        color: #a0a0c0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    .card {
        background: rgba(30, 30, 50, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .stat-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .badge-original {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .badge-cleaned {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-summary {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">VietNews Summarizer</h1>
    <p class="header-subtitle">Hệ thống tóm tắt văn bản tin tức tiếng Việt sử dụng mô hình tối ưu BARTpho</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Mô hình")

    model_name = st.selectbox(
        "Lựa chọn Model Checkpoint",
        [
            "hailinh1509/bartpho-news-summarizer (Fine-tuned)",
            "vinai/bartpho-word (Pre-trained)"
        ]
    )

    max_len = st.slider(
        "Độ dài tối đa tóm tắt (max_length)",
        50, 256, 256, step=10
    )

    min_len = st.slider(
        "Độ dài tối thiểu tóm tắt (min_length)",
        10, 150, 30, step=5
    )

    st.markdown("---")
    st.markdown("### 📊 Trạng thái Dự án")
    st.info(
        "💡 **Hiện tại:** Ứng dụng có thể chọn model Fine-tuned của nhóm "
        "hoặc model Pre-trained gốc từ Hugging Face."
    )

# Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Bài viết đầu vào")

    raw_input = st.text_area(
        "Dán bài báo tiếng Việt cần tóm tắt vào đây:",
        height=320,
        placeholder="Nhập hoặc dán nội dung bài viết tiếng Việt tại đây..."
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    show_cleaned_diff = st.checkbox(
        "Hiển thị so sánh trước và sau khi làm sạch văn bản",
        value=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    btn_summarize = st.button("✨ Tiến hành Tóm tắt", use_container_width=True)

with col2:
    st.markdown("### 📤 Kết quả Tóm tắt & Tiền xử lý")

    if btn_summarize:
        if not raw_input.strip():
            st.warning("⚠️ Vui lòng nhập nội dung bài viết trước khi bấm tóm tắt.")
        else:
            with st.spinner("⏳ Đang thực hiện làm sạch và tóm tắt văn bản..."):
                original_len = len(raw_input.split())

                cleaned_text = clean_text(raw_input)
                cleaned_len = len(cleaned_text.split())

                real_summary = summarize_with_bartpho(
                    cleaned_text,
                    model_name,
                    max_len=max_len,
                    min_len=min_len
                )

                summary_len = len(real_summary.split())

                st.markdown("#### 📝 Văn bản đã tóm tắt:")
                st.success(real_summary)

                st.markdown("#### 📊 Thống kê số từ:")
                st.markdown(f"""
                <span class="stat-badge badge-original">Gốc: {original_len:,} từ</span>
                <span class="stat-badge badge-cleaned">Làm sạch: {cleaned_len:,} từ</span>
                <span class="stat-badge badge-summary">Tóm tắt: {summary_len:,} từ</span>
                """, unsafe_allow_html=True)

                if show_cleaned_diff:
                    with st.expander("🔍 Xem chi tiết văn bản sau khi làm sạch (Cleaned Text)"):
                        st.text_area(
                            "Nội dung sạch:",
                            cleaned_text,
                            height=200,
                            disabled=True
                        )
    else:
        st.info("Nhập văn bản ở cột bên trái và bấm **'Tiến hành Tóm tắt'** để xem kết quả.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>"
    "VietNews Summarizer | Phát triển bởi Antigravity AI Pair Programming | 2026</p>",
    unsafe_allow_html=True
)