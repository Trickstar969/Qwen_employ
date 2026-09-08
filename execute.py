# 终极终极版：transformers原生接口，彻底解决方法缺失问题
from transformers import Qwen2ForCausalLM, AutoTokenizer, GenerationConfig
import torch
import os

# 本地模型缓存路径（复制你的实际路径，加r避免转义）
MODEL_CACHE_PATH = r"D:\Qwen_Model\models\Qwen\Qwen3-8B"
# 验证路径是否存在（确保加载本地模型，不重新下载）
if not os.path.exists(MODEL_CACHE_PATH):
    raise FileNotFoundError(f"本地模型路径不存在：{MODEL_CACHE_PATH}")

# 核心：用transformers原生接口加载分词器和完整生成模型
print("正在加载本地Qwen1.5-0.5B-Chat模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_PATH, trust_remote_code=True)
# Qwen2ForCausalLM：完整的因果语言生成模型，自带generate方法
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_CACHE_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # CPU版用float32，节省内存
    device_map="cpu"  # 强制CPU运行
).eval()  # 推理模式，关闭训练相关层
print("模型加载成功！\n")

# 加载生成配置，控制生成行为
gen_config = GenerationConfig.from_pretrained(MODEL_CACHE_PATH, trust_remote_code=True)
# 初始化多轮对话历史
history = []
# Qwen1.5专属对话模板（保证回复格式正确）
PROMPT_TPL = "{}"

print("Qwen1.5-0.5B-Chat 多轮对话已启动（输入exit退出）：\n")

# 交互式多轮对话循环
while True:
    try:
        # 获取用户输入
        user_input = input("你：")
        if user_input.lower().strip() == "exit":
            print("模型：再见！欢迎下次交流～")
            break
        if not user_input.strip():
            print("模型：请输入有效内容哦～\n")
            continue

        # 拼接对话历史+当前输入
        if not history:
            prompt = PROMPT_TPL.format(user_input)
        else:
            prompt = PROMPT_TPL.format("\n".join(history) + "\n" + user_input)

        # 原生生成逻辑（transformers标准流程，无任何封装）
        with torch.no_grad():  # 关闭梯度，大幅节省CPU内存
            # 编码输入：转tensor，自动放到CPU
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            # 模型生成：用原生generate方法，参数全兼容
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                max_new_tokens=300,  # 最大生成300字，可自行调整
                temperature=0.8,  # 生成随机性，0-1，越小越严谨
                top_p=0.95,  # 核采样，提升回复流畅度
                pad_token_id=tokenizer.eos_token_id,  # 补全符=终止符，避免警告
                eos_token_id=tokenizer.eos_token_id
            )
        # 解码输出：跳过原输入，只取模型生成部分，去除特殊符号
        response = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # 打印并更新历史
        print(f"模型：{response}\n")
        history.append(f"你：{user_input}")
        history.append(f"模型：{response}")

    except Exception as e:
        print(f"模型：抱歉，生成回复时出现小问题：{str(e)[:100]}\n")
        continue