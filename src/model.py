from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# Checkpoint gốc của VinAI
PRETRAINED_CHECKPOINT = "vinai/bartpho-word"

# Checkpoint đã fine-tune của nhóm
FINETUNED_CHECKPOINT = "hailinh1509/bartpho-news-summarizer"


def resolve_checkpoint(model_checkpoint: str = PRETRAINED_CHECKPOINT):
    """
    Chuyển lựa chọn model thành checkpoint thật.
    Vẫn giữ tương thích với code cũ.
    """
    if model_checkpoint in ["finetuned", "fine-tuned", "bartpho-finetuned"]:
        return FINETUNED_CHECKPOINT

    if model_checkpoint in ["pretrained", "pre-trained", "bartpho-pretrained"]:
        return PRETRAINED_CHECKPOINT

    return model_checkpoint


def get_tokenizer(model_checkpoint: str = PRETRAINED_CHECKPOINT):
    """Tải tokenizer cho BARTpho."""
    model_checkpoint = resolve_checkpoint(model_checkpoint)
    return AutoTokenizer.from_pretrained(model_checkpoint)

def get_model(model_checkpoint: str = PRETRAINED_CHECKPOINT):
    """Tải mô hình BARTpho theo loại checkpoint."""
    model_checkpoint = resolve_checkpoint(model_checkpoint)

    if model_checkpoint == PRETRAINED_CHECKPOINT:
        model = AutoModel.from_pretrained(model_checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    model.eval()
    return model


def generate_summary(
    text: str,
    tokenizer,
    model,
    max_length: int = 128,
    min_length: int = 20,
    num_beams: int = 4
):
    """
    Sinh bản tóm tắt bằng BARTpho.

    max_length: độ dài tối đa bản tóm tắt
    min_length: độ dài tối thiểu bản tóm tắt
    num_beams: số nhánh Beam Search
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    )

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def preprocess_tokenize_function(
    examples,
    tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256
):
    """
    Tokenize dữ liệu huấn luyện: content -> input_ids, summary -> labels.
    """

    model_inputs = tokenizer(
        examples["content"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs