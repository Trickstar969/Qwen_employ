
from transformers import Qwen2ForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer
import torch
import os
from transformers.utils import logging
from transformers import set_seed

# 关闭transformers所有冗余警告，彻底无警告运行
logging.set_verbosity_error()
# 设置随机种子（注释则生成结果随机，保留则可复现）
set_seed(42)

# ===================== 自定义配置（仅需修改模型路径！）=====================
# 你的Qwen1.5-1.8B-Chat本地模型路径
MODEL_CACHE_PATH = r"D:\Qwen_Model\models\Qwen\Qwen1___5-1___8B-Chat"
# 1.8B小模型显存占用极低，直接拉满配置（6G GPU即可满配运行）
MAX_CONTEXT_LENGTH = 4096  # Qwen1.5原生支持4096上下文，直接拉满
MAX_NEW_TOKENS = 1024     # 最大生成字数，1.8B可轻松支持1024
# 生成风格参数（按需微调）
TEMPERATURE = 0.7          # 0-1，越小越严谨，越大越发散
TOP_P = 0.95               # 核采样，建议0.9-0.95
REP_PENALTY = 1.05         # 重复惩罚，1.0-1.2，抑制重复回复
# 显存优化（1.8B专属，无需修改，自动适配）
USE_BF16 = True            # bf16更省显存，RTX30/40系/A100建议开启
USE_GRAD_CHECKPOINT = False# 1.8B显存占用极低，关闭以提升生成速度
DEVICE_MAP = "auto"        # 自动设备映射，小显存GPU兜底（6G/8G无压力）
# ==========================================================================

# 验证本地模型路径是否存在
if not os.path.exists(MODEL_CACHE_PATH):
    raise FileNotFoundError(f"本地模型路径不存在，请检查：{MODEL_CACHE_PATH}")

# ===================== GPU/设备自动配置（无需修改）=====================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 动态张量类型：GPU优先bf16→fp16，CPU强制fp32
if DEVICE.type == "cuda":
    if USE_BF16 and torch.cuda.is_bf16_supported():
        TORCH_DTYPE = torch.bfloat16
        dtype_note = "BF16（极致显存优化，6G GPU可跑）"
    else:
        TORCH_DTYPE = torch.float16
        dtype_note = "FP16（显存优化版，全系列GPU适配）"
else:
    TORCH_DTYPE = torch.float32
    dtype_note = "FP32（CPU版）"
    USE_GRAD_CHECKPOINT = False
    DEVICE_MAP = None

# 打印设备信息（已注释显存打印，无API报错）
print("="*60)
print(f"运行设备：{DEVICE.type.upper()} {'(cuda:0)' if DEVICE.type == 'cuda' else ''}")
if DEVICE.type == "cuda":
    print(f"GPU型号：{torch.cuda.get_device_name(0)}")
    # 注释显存打印，避免PyTorch版本API差异报错
    # free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
    # total_mem = torch.cuda.mem_get_info(0)[1] / 1024**3
    # print(f"剩余显存：{free_mem:.1f}G / 总显存：{total_mem:.1f}G")
print(f"模型版本：Qwen1.5-1.8B-Chat | 张量类型：{dtype_note} | 输出模式：流式输出")
print("="*60)

# ===================== 加载分词器+模型（Qwen1.5-Chat专属，GPU加速）=====================
# 1.8B小模型加载极快（10-30秒），无需长时间等待
print("正在加载本地Qwen1.5-1.8B-Chat模型（GPU加速中）...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CACHE_PATH,
    trust_remote_code=True,
    padding_side="right",  # 右填充，符合模型推理逻辑，避免警告
    truncation_side="left" # 左截断，保留最新对话内容，防止上下文溢出
)

# 修复核心2：移除gradient_checkpointing参数，改用专用方法设置
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_CACHE_PATH,
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    device_map=DEVICE_MAP,
).eval()

# 正确设置梯度检查点（Qwen1.5系列专用方法）
if USE_GRAD_CHECKPOINT:
    model.gradient_checkpointing_enable()
else:
    model.gradient_checkpointing_disable()

# 非自动映射时，手动将模型移到指定设备（保持兼容性）
if DEVICE_MAP is None:
    model = model.to(DEVICE)

print("✅ Qwen1.5-1.8B-Chat模型加载成功！\n")

# ===================== 加载官方生成配置（彻底消除弃用警告）=====================
gen_config = GenerationConfig.from_pretrained(MODEL_CACHE_PATH, trust_remote_code=True)
# 覆盖生成参数（所有参数统一管理，无任何弃用警告）
gen_config.max_new_tokens = MAX_NEW_TOKENS
gen_config.temperature = TEMPERATURE
gen_config.top_p = TOP_P
gen_config.repetition_penalty = REP_PENALTY
gen_config.do_sample = True  # 开启采样，保证temperature/top_p生效（关键）
gen_config.pad_token_id = tokenizer.eos_token_id  # 补全符=终止符
gen_config.eos_token_id = tokenizer.eos_token_id  # 终止符配置
gen_config.max_length = MAX_CONTEXT_LENGTH        # 最大上下文长度，防止溢出

# ===================== 多轮对话核心配置（Qwen1.5-Chat官方原生模板）=====================
chat_history = []  # 对话历史：[{"role": "user/assistant", "content": ...}, ...]

# ===================== 工具函数：清理CUDA显存（避免泄漏）=====================
def clear_cuda_cache():
    """GPU模式下清理无用显存，多轮对话不累积"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ===================== 交互式多轮对话（流式输出+官方Chat模板）=====================
print("🚀 Qwen1.5-1.8B-Chat (GPU加速+流式输出) 多轮对话已启动")
print("📌 指令：输入 exit/退出/q 结束对话 | 输入 clear/清空/重置 清空对话历史")
print("="*40 + "\n")

while True:
    try:
        # 获取用户输入并清理空格
        user_input = input("你：").strip()
        # 指令1：退出对话
        if user_input.lower() in ["exit", "退出", "q"]:
            print("模型：再见！欢迎下次和我交流～ 😊")
            clear_cuda_cache()
            break
        # 指令2：清空对话历史
        if user_input.lower() in ["clear", "清空", "重置"]:
            chat_history = []
            print("模型：已为你清空所有对话历史，现在可以开始新的对话啦～\n")
            clear_cuda_cache()
            continue
        # 过滤空输入
        if not user_input:
            print("模型：请输入有效内容哦，我会认真回复你的～\n")
            continue

        # 1. 构建Qwen1.5-Chat官方原生Prompt（Chat模型专属，最标准）
        prompt = tokenizer.apply_chat_template(
            conversation=chat_history + [{"role": "user", "content": user_input}],
            tokenize=False,               # 不直接编码，先生成字符串prompt
            add_generation_prompt=True,   # 自动添加assistant生成前缀，必须开启
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS  # 预留生成空间，防止溢出
        )

        # 2. 编码输入：转tensor+移到GPU/CPU+自动截断
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS,
                padding=False
            ).to(DEVICE)  # 输入张量与模型设备严格对齐

        # 3. 初始化流式输出器（实时打字效果，参数最优配置）
        streamer = TextStreamer(
            tokenizer,
            skip_special_tokens=True,    # 跳过<|im_start|>等特殊符号
            skip_prompt=True,            # 跳过原始prompt，只打印生成内容
            clean_up_tokenization_spaces=True,  # 清理多余空格
            flush=True                   # 强制实时刷新，无延迟
        )

        # 4. 模型生成+流式输出（核心逻辑，无警告）
        print("模型：", end="", flush=True)  # 固定模型前缀，格式统一
        with torch.no_grad():  # 关闭梯度，节省90%以上显存
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                streamer=streamer,  # 传入流式输出器，实现实时打字
                use_cache=True,     # 开启缓存，提升多轮对话生成速度
            )

        # 5. 解码获取完整回复（用于更新对话历史，多轮对话核心，不能删！）
        response = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        # 6. 更新对话历史+清理显存
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        clear_cuda_cache()
        print("\n")  # 换行分隔，提升对话可读性

    # 专属异常：GPU显存不足（1.8B极少出现，出现则按提示微调）
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ 模型：GPU显存不足！1.8B专属解决方案（极简）：")
            print("  1. 减小MAX_CONTEXT_LENGTH（如改为2048）")
            print("  2. 减小MAX_NEW_TOKENS（如改为512）")
        else:
            print(f"\n❌ 模型：运行错误：{str(e)[:150]}")
        clear_cuda_cache()
        print()
        continue
    # 通用异常处理
    except Exception as e:
        print(f"\n❌ 模型：抱歉，生成回复时出现小问题：{str(e)[:150]}\n")
        clear_cuda_cache()
        continue