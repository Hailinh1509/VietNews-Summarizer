from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_tokenizer(model_checkpoint: str = "vinai/bartpho-word"):
    """Tải và khởi tạo tokenizer cho BARTpho."""
    return AutoTokenizer.from_pretrained(model_checkpoint)

def get_model(model_checkpoint: str = "vinai/bartpho-word"):
    """Tải mô hình Seq2Seq BARTpho."""
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

def preprocess_tokenize_function(examples, tokenizer, max_input_length: int = 1024, max_target_length: int = 256):
    """
    Thực hiện tokenize (chuyển chữ sang số), đồng thời giới hạn độ dài (truncation) 
    và thêm padding về độ dài tối đa cho cả đầu vào (content) và nhãn (summary).
    """
    # Tokenize phần nội dung bài viết
    model_inputs = tokenizer(
        examples["content"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )

    # Tokenize phần tóm tắt
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )

    # Gán nhãn cho mô hình huấn luyện
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
