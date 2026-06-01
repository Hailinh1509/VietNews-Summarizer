import streamlit as st
import sys
import os
import time

# Thêm thư mục gốc vào path để có thể import từ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import clean_text

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="VietNews Summarizer - AI Tóm tắt Tin tức",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thiết lập Style CSS tùy chỉnh cho ứng dụng trông hiện đại và chuyên nghiệp (Premium Dark/Glassmorphism theme)
st.markdown("""
<style>
    /* Nhập font chữ Outfit và Inter từ Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    /* Thiết lập font chữ chung */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Gradient Background cho Header */
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
    
    /* Glassmorphism Card Style */
    .card {
        background: rgba(30, 30, 50, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Custom Stats badges */
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

# Giao diện chính Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">VietNews Summarizer</h1>
    <p class="header-subtitle">Hệ thống tóm tắt văn bản tin tức tiếng Việt sử dụng mô hình tối ưu BARTpho</p>
</div>
""", unsafe_allow_html=True)

# Sidebar cấu hình mô hình
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Mô hình")
    model_name = st.selectbox(
        "Lựa chọn Model Checkpoint",
        ["vinai/bartpho-word (Fine-tuned)", "vinai/bartpho-word (Pre-trained)"]
    )
    
    max_len = st.slider("Độ dài tối đa tóm tắt (max_length)", 50, 512, 256, step=10)
    min_len = st.slider("Độ dài tối thiểu tóm tắt (min_length)", 10, 150, 30, step=5)
    
    st.markdown("---")
    st.markdown("### 📊 Trạng thái Dự án")
    st.info("💡 **Hiện tại:** Prototype đang sử dụng mô hình tóm tắt mẫu (mockup) kết hợp bộ tiền xử lý `src.preprocess.clean_text` thực tế. Khi quá trình fine-tuning hoàn tất, file checkpoint sẽ được tích hợp trực tiếp tại đây.")

# Layout chia làm 2 cột: Cột trái (Nhập liệu và làm sạch), Cột phải (Kết quả tóm tắt)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Bài viết đầu vào")
    raw_input = st.text_area(
        "Dán bài báo tiếng Việt cần tóm tắt vào đây:",
        height=320,
        placeholder="Nhập hoặc dán nội dung bài viết tiếng Việt tại đây (bao gồm cả các thẻ HTML, link liên kết, email... hệ thống sẽ tự động dọn dẹp)..."
    )
    
    # Checkbox cấu hình các bước làm sạch bổ sung
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    show_cleaned_diff = st.checkbox("Hiển thị so sánh trước và sau khi làm sạch văn bản", value=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    btn_summarize = st.button("✨ Tiến hành Tóm tắt", use_container_width=True)

with col2:
    st.markdown("### 📤 Kết quả Tóm tắt & Tiền xử lý")
    
    if btn_summarize:
        if not raw_input.strip():
            st.warning("⚠️ Vui lòng nhập nội dung bài viết trước khi bấm tóm tắt.")
        else:
            with st.spinner("⏳ Đang thực hiện làm sạch và tóm tắt văn bản..."):
                # 1. Đo lường các chỉ số trước dọn dẹp
                original_len = len(raw_input)
                
                # 2. Gọi hàm tiền xử lý từ src.preprocess
                cleaned_text = clean_text(raw_input)
                cleaned_len = len(cleaned_text)

                
                # 3. Tạo mockup tóm tắt (giả định)
                time.sleep(1.0) # Tạo hiệu ứng xử lý AI
                
                # Mockup tóm tắt đơn giản bằng cách trích xuất 2 câu đầu tiên hoặc tóm tắt giả định
                sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
                if len(sentences) >= 2:
                    mock_summary = f"**Bản tóm tắt (BARTpho Mockup):**\n\n{sentences[0]}. {sentences[1]}."
                elif sentences:
                    mock_summary = f"**Bản tóm tắt (BARTpho Mockup):**\n\n{sentences[0]}."
                else:
                    mock_summary = "Không thể tạo bản tóm tắt từ văn bản đã nhập."
                
                summary_len = len(mock_summary)
                
                # Trực quan hóa kết quả
                st.markdown("#### 📝 Văn bản đã tóm tắt:")
                st.success(mock_summary)
                
                # Thống kê độ dài
                st.markdown("#### 📊 Thống kê ký tự:")
                st.markdown(f"""
                <span class="stat-badge badge-original">Gốc: {original_len:,} ký tự</span>
                <span class="stat-badge badge-cleaned">Làm sạch: {cleaned_len:,} ký tự</span>
                <span class="stat-badge badge-summary">Tóm tắt: {summary_len:,} ký tự</span>
                """, unsafe_allow_html=True)
                
                # Hiển thị text đã làm sạch nếu được tích chọn
                if show_cleaned_diff:
                    with st.expander("🔍 Xem chi tiết văn bản sau khi làm sạch (Cleaned Text)"):
                        st.text_area("Nội dung sạch:", cleaned_text, height=200, disabled=True)
    else:
        st.info("Nhập văn bản ở cột bên trái và bấm **'Tiến hành Tóm tắt'** để xem kết quả.")

# Footer của ứng dụng
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>"
    "VietNews Summarizer | Phát triển bởi Antigravity AI Pair Programming | 2026</p>",
    unsafe_allow_html=True
)
