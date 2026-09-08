from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 换成你实际用的模型路径/名称
model_name_or_path = r"D:\Qwen_Model\models\Qwen\Qwen3___5-2B"  # 或你的本地路径

# 加载模型（多模态）
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 打印模型整体结构（看模块）
print("=== 模型顶层结构 ===")
for name, module in model.named_children():
    print(f"{name}: {type(module).__name__}")

# 打印前几层注意力层的具体参数名（你最关心的）
print("\n=== 前2层注意力层的参数名 ===")
for i, layer in enumerate(model.model.layers[:2]):
    print(f"\n--- 第 {i} 层 ---")
    for name, param in layer.named_parameters():
        if any(kw in name for kw in ["qkv", "proj", "attn"]):
            print(f"  {name}: {param.shape}")