import re
import unicodedata
import pandas as pd

def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về dạng NFC (quan trọng với tiếng Việt)."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)

def remove_html_tags(text: str) -> str:
    """Xóa thẻ HTML còn sót lại."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", " ", text) #[^>] - mọi ký tự khác dấu >

def remove_urls(text: str) -> str:
    """Xóa URL."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def remove_email(text: str) -> str:
    """Xóa địa chỉ email."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\S+@\S+\.\S+", " ", text)

def remove_extra_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng: nhiều dấu cách/tab/newline → 1 dấu cách."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[ \t]+", " ", text)        # nhiều dấu cách/tab → 1
    text = re.sub(r"\n{2,}", "\n", text)        # nhiều newline → 1
    return text.strip()

def remove_special_chars(text: str) -> str:
    """
    Xóa các ký tự đặc biệt không liên quan:
      - Emoji, ký hiệu hình học, ký tự điều khiển, v.v.
    Giữ lại: chữ cái (Latin + tiếng Việt), số, dấu câu cơ bản.
    """
    if not isinstance(text, str):
        return ""
    # Xóa ký tự điều khiển (trừ newline \n)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Xóa emoji và ký hiệu ngoài BMP
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    # Xóa các chuỗi ký hiệu lặp vô nghĩa (ví dụ: >>>>, ====, ****)
    text = re.sub(r"([><=\-\*#\|~^])\1{2,}", " ", text)
    return text

def remove_author_byline(text: str) -> str:
    """
    Xóa tên tác giả / byline thường xuất hiện cuối bài báo tiếng Việt.
    Ví dụ: 'Lê Đăng.', 'Huy Anh (Vietnam+).', 'Phúc Duy.'
    Pattern: 1-4 từ, kết thúc bằng dấu chấm, ở cuối chuỗi.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"\n([A-ZÁÀẢÃẠĂẮẶẲẴẤẦẨẪẬÂÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
        r"A-Za-zÀ-ỹ\s\.]{3,50})\.\s*$",
        "",
        text,
        flags=re.MULTILINE
    )
    return text.strip()

def remove_image_captions(text: str) -> str:
    """Xóa chú thích ảnh thường gặp trong bài viết."""
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"\s*(Ảnh minh họa|Ảnh|Nguồn|Hình ảnh)\s*:\s*[^.\n]{1,80}\.",
        "",
        text,
        flags=re.IGNORECASE
    )
    return text

def remove_source_lines(text: str) -> str:
    """Xóa dòng ghi nguồn bài viết ở cuối."""
    if not isinstance(text, str):
        return ""
    pattern = (
        r"^\s*\(?(Theo|Nguồn)\s+"
        r"[A-ZA-Za-zÀ-ỹ0-9\s\.\-\+]{2,50}"
        r"\)?\.?\s*$"
    )
    lines = text.split("\n")
    cleaned_lines = [
        line for line in lines
        if not re.match(pattern, line.strip(), flags=re.IGNORECASE)
    ]
    return "\n".join(cleaned_lines)

def remove_video_invitation(text: str) -> str:
    """Xóa lời mời xem video thường thấy cuối bài."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"Mời độc giả xem thêm video[^\n]*", "", text, flags=re.IGNORECASE)
    return text

def remove_related_news_links(text: str) -> str:
    """Xóa các dòng '>> Tiêu đề bài liên quan' thường ở đầu bài."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r">>.*", "", text)
    return text

def normalize_punctuation(text: str) -> str:
    """Chuẩn hóa dấu câu: dấu lặp, dấu thừa."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\.{2,}", ".", text)          # .... → .
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r",{2,}", ",", text)
    return text

# gọi lần lượt các hàm làm sạch đã xây dựng trước đó để biến một văn bản thô thành văn bản sạch, 
# sẵn sàng cho bước tokenization và fine-tuning BARTPho.
def clean_text(text: str) -> str:
    """Pipeline làm sạch hoàn chỉnh cho một chuỗi văn bản."""
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_email(text)
    text = remove_image_captions(text)
    text = remove_video_invitation(text)
    text = remove_related_news_links(text)
    text = remove_author_byline(text)
    text = remove_source_lines(text)
    text = remove_special_chars(text)
    text = normalize_punctuation(text)
    text = remove_extra_whitespace(text)
    return text

def filter_df(
    df: pd.DataFrame,
    min_content_len: int = 100,
    max_content_len: int = 6000,
    min_summary_len: int = 30,
    max_summary_len: int = 1000
) -> pd.DataFrame:
    """
    Lọc bỏ các mẫu không hợp lệ trong DataFrame dữ liệu:
    - Xóa các dòng null ở content hoặc summary.
    - Lọc theo giới hạn độ dài ký tự của content và summary.
    - Loại bỏ các dòng bị trùng lặp content.
    - Loại bỏ mẫu có summary dài hơn hoặc bằng content.
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    n0 = len(df)
    # Xóa null / rỗng
    df = df.dropna(subset=["content", "summary"])
    df = df[df["content"].str.strip() != ""]
    df = df[df["summary"].str.strip() != ""]

    # Lọc theo độ dài
    c_len = df["content"].str.len()
    s_len = df["summary"].str.len()
    df = df[
        (c_len >= min_content_len) & (c_len <= max_content_len) &
        (s_len >= min_summary_len) & (s_len <= max_summary_len)
    ]

    # Loại bỏ trùng lặp (content giống hệt nhau)
    df = df.drop_duplicates(subset=["content"])

    # Loại bỏ mẫu có summary dài hơn / bằng content (tóm tắt không hợp lý)
    df = df[df["summary"].str.len() < df["content"].str.len()]

    print(f"  Trước lọc: {n0:,} | Sau lọc: {len(df):,} "
          f"| Đã loại: {n0 - len(df):,} ({(n0-len(df))/n0*100:.1f}%)")
    return df.reset_index(drop=True)
