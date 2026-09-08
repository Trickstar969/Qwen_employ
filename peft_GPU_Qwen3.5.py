import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from threading import Thread
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
model_path = r"D:\Qwen_Model\models\Qwen\Qwen3___5-0___8B"
lora_path = "qwen3.5-lora"  # 你要用的纯LoRA

# ==================== 加载模型 ====================
print("正在加载模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cuda:0"
)

# ------------- 加载 LoRA -------------
model = PeftModel.from_pretrained(model, lora_path)
model = model.to("cuda:0")
model.eval()

# ==================== 【LoRA 完整信息展示】 ====================
print("\n" + "="*60)
print("🔍 LoRA 真实挂载检测 + 可训练参数详情")
print("="*60)

is_peft_model = isinstance(model, PeftModel)
has_peft_config = hasattr(model, "peft_config")
has_lora_params = any("lora_" in name for name, param in model.named_parameters())

print(f"📌 是否为 LoRA 模型: {is_peft_model}")
print(f"📌 是否加载 LoRA 配置: {has_peft_config}")
print(f"📌 检测到 LoRA 权重层: {has_lora_params}")

# --------------------- 【新加：显示所有 LoRA 可训练参数】 ---------------------
if has_lora_params:
    print("\n📊 LoRA 可训练参数列表：")
    lora_param_names = [name for name, param in model.named_parameters() if "lora_" in name]
    for name in lora_param_names[:20]:  # 显示前20个，避免刷屏
        print(f"   → {name}")
    if len(lora_param_names) > 20:
        print(f"   ... 共 {len(lora_param_names)} 个 LoRA 参数")

    # 统计可训练参数量
    total_trainable = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n⚙️  总模型参数: {total_params:,}")
    print(f"🧩 LoRA 可训练参数: {total_trainable:,}")
    print(f"📈 可训练比例: {total_trainable / total_params * 100:.4f}%")

# 最终状态
if is_peft_model and has_peft_config and has_lora_params:
    print("\n✅ ✅ ✅ LoRA 已真正成功挂载！")
else:
    print("\n❌ ❌ ❌ LoRA 未成功挂载！")
print("="*60 + "\n")

# ==================== 无限循环 ====================
while True:
    print("="*60)
    print("新一轮对话（输入 q 退出）")
    print("="*60)

    image_path = input("请输入图片路径（直接回车=纯文本）：").strip().strip('"').strip("'")
    if image_path.lower() == "q":
        break

    prompt = input("请输入你的问题：").strip()
    if prompt.lower() == "q":
        break

    if image_path == "":
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        images = None
    else:
        try:
            image = Image.open(image_path)
            images = [image]
        except Exception as e:
            print("图片打开失败：", e)
            continue

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except:
        text = prompt

    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    print("\n模型输出：", end="", flush=True)
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for text in streamer:
        print(text, end="", flush=True)

    print("\n")

print("程序已退出")