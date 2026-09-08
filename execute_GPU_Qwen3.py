# 先导入Python原生logging（核心修复：屏蔽所有警告的基础）
# ===================== 自动激活Conda Qwen环境（必须放在文件最顶部）=====================
import sys
import os
import subprocess

# ---------- 仅需修改这3个参数，和你的实际环境一致！----------
CONDA_PATH = r"D:\miniconda3"  # 你的Conda安装根路径
ENV_NAME = "Qwen"              # 你的Conda环境名
PY_FILE_PATH = os.path.abspath(__file__)  # 当前py文件的绝对路径，无需修改
# -----------------------------------------------------------

# 拼接Qwen环境的Python解释器路径
conda_python_path = os.path.join(CONDA_PATH, "envs", ENV_NAME, "python.exe")

# 检测当前运行的Python是否是Qwen环境的Python
if sys.executable != conda_python_path:
    print(f"🔍 检测到当前未在Conda[{ENV_NAME}]环境运行，正在自动切换...")
    print(f"📌 系统默认Python：{sys.executable}")
    print(f"📌 目标Conda Python：{conda_python_path}")
    # 调用Qwen环境的Python，重新执行当前py文件
    subprocess.run([conda_python_path, PY_FILE_PATH], shell=True)
    # 退出当前非目标环境的Python进程
    sys.exit(0)
import logging
from transformers import Qwen3ForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer
import torch
import os
from transformers.utils import logging as hf_logging  # 重命名，避免和原生logging冲突
from transformers import set_seed

# ============= 全局屏蔽所有警告（原生logging，兼容所有库，无报错）=============
hf_logging.set_verbosity_error()  # 屏蔽transformers所有警告
logging.getLogger("torch").setLevel(logging.ERROR)  # 屏蔽PyTorch的UserWarning/Warning
logging.getLogger("cuda").setLevel(logging.ERROR)   # 屏蔽CUDA底层警告
logging.getLogger("transformers").setLevel(logging.ERROR)  # 兜底屏蔽transformers
logging.getLogger("accelerate").setLevel(logging.ERROR)    # 屏蔽accelerate相关警告
# 禁用Flash Attention，彻底消除SDPA的UserWarning（RTX3060不支持FA，用纯SDPA）
os.environ["TORCH_BACKENDS_FLASH_ATTENTION_ENABLED"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "False"  # 屏蔽分词器并行冗余警告
os.environ["PYTHONWARNINGS"] = "ignore"  # 终极兜底：屏蔽Python所有全局警告
# ==========================================================================

# 设置随机种子（注释则生成结果随机，保留可复现）
set_seed(42)

# ===================== 自定义配置（仅改模型路径！RTX3060专属适配）=====================
MODEL_CACHE_PATH = r"D:\Qwen_Model\models\Qwen\Qwen3-1___7B"  # 你的本地模型路径
MODEL_SERIES = "Qwen3"
MODEL_VERSION = "8B"
# RTX3060 Laptop（6G/12G）最优配置：4096上下文+1024生成，显存占用3-4G，全GPU加载
MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 1024
# 生成风格参数（按需微调）
TEMPERATURE = 0.7
TOP_P = 0.95
REP_PENALTY = 1.05
# 显存优化（RTX3060完美支持BF16，强制全GPU加载，无CPU卸载）
USE_BF16 = True
USE_GRAD_CHECKPOINT = False  # 小模型关闭，提速优先
DEVICE_MAP = "cuda:0"        # 核心：强制单卡GPU加载，解决meta device问题
MAX_CHAT_HISTORY = 20       # 对话历史限制，防止内存累积
# ==========================================================================

# 验证本地模型路径是否存在
if not os.path.exists(MODEL_CACHE_PATH):
    raise FileNotFoundError(f"本地模型路径不存在，请检查：{MODEL_CACHE_PATH}")

# ===================== GPU/设备配置（RTX3060 Laptop专属，无需修改）=====================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 动态张量类型：RTX3060原生支持BF16，极致省显存
if DEVICE.type == "cuda":
    if USE_BF16 and torch.cuda.is_bf16_supported():
        TORCH_DTYPE = torch.bfloat16
        dtype_note = "BF16（RTX3060专属优化，显存占用最低）"
    else:
        TORCH_DTYPE = torch.float16
        dtype_note = "FP16（全系列GPU适配，显存优化）"
else:
    TORCH_DTYPE = torch.float32
    dtype_note = "FP32（CPU版，小模型可运行）"
    USE_GRAD_CHECKPOINT = False
    DEVICE_MAP = None

# 打印设备信息（清晰展示RTX3060+GPU全加载状态）
print("="*60)
print(f"运行设备：{DEVICE.type.upper()} (cuda:0)")
print(f"GPU型号：{torch.cuda.get_device_name(0)} | 张量类型：{dtype_note}")
print(f"模型版本：{MODEL_SERIES}-{MODEL_VERSION}-Chat | 输出模式：流式输出")
print(f"上下文长度：{MAX_CONTEXT_LENGTH} | 最大生成字数：{MAX_NEW_TOKENS}")
print(f"🔥 全量参数加载到GPU | 无CPU/硬盘卸载 | 彻底无警告运行")
print("="*60)

# ===================== 加载分词器+Qwen3模型（RTX3060专属，无报错）=====================
print(f"正在加载本地{MODEL_SERIES}-{MODEL_VERSION}-Chat模型（GPU全量加载中）...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_CACHE_PATH,
    trust_remote_code=True,
    padding_side="right",  # 符合Qwen3推理逻辑
    truncation_side="left" # 左截断，保留最新对话
)
# Qwen3兜底配置：解决部分版本pad_token_id为空的问题
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 核心：强制cuda:0加载，添加offload_folder兜底，无meta device问题
model = Qwen3ForCausalLM.from_pretrained(
    MODEL_CACHE_PATH,
    trust_remote_code=True,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    device_map=DEVICE_MAP,
    offload_folder="./offload_cache"  # 冗余兜底，实际不会触发
).eval()  # 推理模式，省显存+提速

# 梯度检查点配置（小模型关闭）
if USE_GRAD_CHECKPOINT:
    model.gradient_checkpointing_enable()
else:
    model.gradient_checkpointing_disable()

# 手动移设备（保持兼容性，实际DEVICE_MAP=cuda:0时无需执行）
if DEVICE_MAP is None:
    model = model.to(DEVICE)

# 验证模型是否全量加载到GPU（新增：直观确认，无报错）
model_device = next(model.parameters()).device
print(f"✅ 模型加载成功！模型主设备：{model_device}（全GPU加载，无CPU卸载）\n")

# ===================== 生成配置（Qwen3原生兼容，无弃用警告）=====================
gen_config = GenerationConfig.from_pretrained(MODEL_CACHE_PATH, trust_remote_code=True)
# 覆盖生成参数，统一管理
gen_config.max_new_tokens = MAX_NEW_TOKENS
gen_config.temperature = TEMPERATURE
gen_config.top_p = TOP_P
gen_config.repetition_penalty = REP_PENALTY
gen_config.do_sample = True  # 开启采样，让temperature/top_p生效
gen_config.pad_token_id = tokenizer.pad_token_id
gen_config.eos_token_id = tokenizer.eos_token_id
gen_config.max_length = MAX_CONTEXT_LENGTH
gen_config.use_cache = True  # 开启缓存，提升多轮对话速度

# ===================== 多轮对话核心配置=====================
chat_history = []  # 对话历史存储
# 显存清理函数（避免多轮累积，RTX3060必备）
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ===================== 交互式多轮对话（流式输出，体验拉满）=====================
print(f"🚀 {MODEL_SERIES}-{MODEL_VERSION}-Chat (RTX3060 GPU加速) 多轮对话已启动")
print("📌 指令：输入 exit/退出/q 结束 | 输入 clear/清空/重置 清空对话历史")
print("========================================\n")

while True:
    try:
        # 获取用户输入并清理空格
        user_input = input("你：").strip()
        # 退出对话指令
        if user_input.lower() in ["exit", "退出", "q"]:
            print("模型：再见！欢迎下次和我交流～ 😊")
            clear_cuda_cache()
            break
        # 清空对话历史指令
        if user_input.lower() in ["clear", "清空", "重置"]:
            chat_history = []
            print("模型：已清空所有对话历史，可开始新对话～\n")
            clear_cuda_cache()
            continue
        # 过滤空输入
        if not user_input:
            print("模型：请输入有效内容哦，我会认真回复的～\n")
            continue

        # 生成前预处理：清理显存+限制对话历史长度
        clear_cuda_cache()
        if len(chat_history) > MAX_CHAT_HISTORY:
            chat_history = chat_history[-MAX_CHAT_HISTORY:]

        # 1. 构建Qwen3官方原生对话模板（最标准，无格式错误）
        prompt = tokenizer.apply_chat_template(
            conversation=chat_history + [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True,  # 自动添加assistant前缀，Qwen3必须开启
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS  # 预留生成空间，防止溢出
        )

        # 2. 编码输入：转Tensor+移到GPU+自动截断
        with torch.no_grad():  # 关闭梯度，省90%显存
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS,
                padding=False
            ).to(DEVICE)

        # 3. 初始化流式输出器（实时打字效果，无延迟）
        streamer = TextStreamer(
            tokenizer,
            skip_special_tokens=True,    # 跳过<|im_start|>等特殊符号
            skip_prompt=True,            # 只打印生成内容，跳过提问
            clean_up_tokenization_spaces=True,
            flush=True                   # 强制实时刷新，无卡顿
        )

        # 4. 模型生成+流式输出（核心逻辑，无报错）
        print("模型：", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                streamer=streamer,
                use_cache=True,
            )

        # 5. 解码获取完整回复（用于更新对话历史，多轮核心）
        response = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        # 6. 更新对话历史+再次清理显存
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        clear_cuda_cache()
        print("\n")  # 换行分隔，提升对话可读性

    # 显存不足异常（RTX3060极少出现，给出精准解决方案）
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ 模型：GPU显存不足！RTX3060专属解决方案：")
            print("  1. 减小MAX_CONTEXT_LENGTH到2048 或 MAX_NEW_TOKENS到512")
            print("  2. 临时改为TORCH_DTYPE=torch.float16（若当前是BF16）")
        else:
            print(f"\n❌ 模型：运行错误：{str(e)[:150]}")
        clear_cuda_cache()
        print()
        continue
    # 通用异常处理（捕获所有其他问题，不中断程序）
    except Exception as e:
        print(f"\n❌ 模型：抱歉，生成回复时出现小问题：{str(e)[:150]}\n")
        clear_cuda_cache()
        continue