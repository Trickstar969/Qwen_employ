import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from threading import Thread
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
model_path = r"D:\Qwen_Model\models\Qwen\Qwen3___5-2B"

# ==================== 加载模型 ====================
print("正在加载模型...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cuda:0"
)
print("模型加载完成！\n")

# ==================== 无限循环 ====================
while True:
    print("=" * 60)
    print("新一轮对话（输入 q 退出）")
    print("=" * 60)

    # 输入图片路径：直接回车 = 纯文本对话
    image_path = input("请输入图片路径（直接回车=纯文本）：").strip().strip('"').strip("'")
    if image_path.lower() == "q":
        break

    # 输入问题
    prompt = input("请输入你的问题：").strip()
    if prompt.lower() == "q":
        break

    # ============== 核心：区分纯文本 / 图文 ==============
    if image_path == "":
        # 纯文本对话
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        images = None
    else:
        # 图文模式
        try:
            image = Image.open(image_path)
            images = [image]
        except Exception as e:
            print("图片打开失败：", e)
            continue

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

    # 构造输入
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt"
    ).to(model.device)

    # 流式输出
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