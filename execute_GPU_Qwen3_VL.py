import torch
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import TextIteratorStreamer
from threading import Thread

# ===================== 模型路径 =====================
MODEL_PATH = r"D:\Qwen_Model\models\Qwen\Qwen3-VL-4B-Instruct"

# 加载模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ===================== 🔥 键盘输入 =====================
# 输入图片路径
image_path = input("请输入图片路径：").strip().strip('"').strip("'")  # 自动去掉引号

# 输入问题
question = input("请输入你的问题：").strip()

# 构造消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]
    }
]

# ===================== 处理输入 =====================
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# ===================== 流式输出 =====================
streamer = TextIteratorStreamer(
    processor.tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

gen_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": 1024}

print("\n模型：", end="", flush=True)
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)

print()