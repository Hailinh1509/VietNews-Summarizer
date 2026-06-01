from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_tokenizer(model_checkpoint: str = "vinai/bartpho-word"):
    """Tải và khởi tạo tokenizer cho BARTpho."""
    return AutoTokenizer.from_pretrained(model_checkpoint)

def get_model(model_checkpoint: str = "vinai/bartpho-word"):
    """Tải mô hình Seq2Seq BARTpho."""
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
