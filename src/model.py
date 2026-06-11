from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    model.eval()
    return model


def generate_summary(
    text: str,
    tokenizer,
    model,
    max_new_tokens: int = 256,
    min_length: int = 30,
    num_beams: int = 4
):
    """
    Sinh bản tóm tắt bằng mô hình BARTPho.

    Mục đích:
    - Dùng chung cho cả Pre-trained và Fine-tuned.
    - Văn bản đầu vào được tokenizer chuyển thành input_ids và attention_mask.
    - Model dùng generate() để sinh output.
    - Cấu hình generate được đặt giống phần test trong notebook fine-tuning:
      max_new_tokens=256, min_length=30, num_beams=4.
    """

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
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
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