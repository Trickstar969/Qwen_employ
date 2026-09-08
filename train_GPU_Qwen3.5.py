import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ==================== 配置 ====================
model_path = r"D:\Qwen_Model\models\Qwen\Qwen3___5-0___8B"
output_dir = "./qwen3.5-lora"
max_seq_len = 2048
num_epochs = 1
learning_rate = 1e-4

# ==================== 加载模型 ====================
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# ==================== LoRA 正确配置 ====================
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== 训练数据 ====================
train_data = [
    {
        "image": r"C:\Users\1\Desktop\20.png",
        "text": "这是什么？",
        "answer": "这是阿帕奇武装直升机。"
    }
]

# ==================== 数据预处理 ====================
def process_fn(item):
    image = Image.open(item["image"]).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": item["text"]}
        ]},
        {"role": "assistant", "content": item["answer"]}
    ]

    inputs = processor(
        text=processor.apply_chat_template(messages, tokenize=False),
        images=image,
        return_tensors="pt",
        max_length=max_seq_len,
        padding="max_length",
        truncation=False
    )

    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

dataset = Dataset.from_list(train_data).map(process_fn)

# ==================== 训练参数 ====================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=learning_rate,
    logging_steps=1,
    save_strategy="epoch",            # 按epoch保存
    save_total_limit=3,               # 最多保留3个checkpoint
    optim="adamw_torch",
    bf16=True,
    report_to="none",
    remove_unused_columns=False,      # 多模态必须加
)

# ==================== 训练 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# ------------------------------------------------------------------------------
# ✅ ✅ ✅ 【你要的：生成 LoRA Checkpoint / 纯 Adapter 保存】
# ------------------------------------------------------------------------------
# 1. 保存最终 LoRA 权重（纯 adapter，非完整模型）
model.save_pretrained("qwen3.5-lora-final")

# 2. 也保存一份在 trainer 的输出目录（和中间ckpt在一起）
model.save_pretrained(f"{output_dir}/final_lora")

# 3. 打印保存路径
print("\n" + "="*60)
print("✅ 训练完成！LoRA Adapter 已保存：")
print("📁 qwen3.5-lora-final/")
print("📁", f"{output_dir}/final_lora/")
print("="*60)