import os
import numpy as np
from PIL import Image
import mindtorch
import mindhf
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info  # 请确保该模块在你的环境可用
from transformers import AutoModel
import gradio as gr
from argparse import ArgumentParser
import copy
import requests
from io import BytesIO
import tempfile
import hashlib
import gc

# 关键优化：设置环境变量加速 transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer警告
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def _get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="lvyufeng/HunyuanOCR",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )

    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        default=False,
        help="Enable flash_attention_2 when loading the model.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )

    args = parser.parse_args()
    return args


def _load_model_processor(args):
    print(f"[INFO] 加载模型（eager 模式）")

    model = AutoModel.from_pretrained(
        args.checkpoint_path,
        attn_implementation="eager",
        torch_dtype=mindtorch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 关键：禁用梯度检查点（如果启用会导致极慢）
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        print(f"[INFO] 梯度检查点已禁用")

    # 设置为评估模式
    model.eval()
    print(f"[INFO] 模型设置为评估模式")

    processor = AutoProcessor.from_pretrained(
        args.checkpoint_path, use_fast=False, trust_remote_code=True
    )

    print(f"[INFO] 模型加载完成，当前设备: {next(model.parameters()).device}")
    return model, processor


def _parse_text(text):
    """解析文本，处理特殊格式"""
    # if text is None:
    #     return text
    text = text.replace("<trans>", "").replace("</trans>", "")
    return text


def _remove_image_special(text):
    """移除图像特殊标记"""
    # if text is None:
    #     return text
    # # 移除可能的图像特殊标记
    # import re
    # text = re.sub(r'<image>|</image>|<img>|</img>', '', text)
    # return text
    return text


def _gc():
    """垃圾回收"""
    gc.collect()
    if mindtorch.cuda.is_available():
        mindtorch.cuda.empty_cache()


def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n < 2000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[: n - length * (count - 1)]

    return text


def _launch_demo(args, model, processor):
    # 全局变量用于跟踪是否是首次调用
    first_call = [True]

    # 关键修复：移除 model 和 processor 参数，使用闭包访问
    # 增加 duration 到 120 秒，避免高峰期超时
    def call_local_model(messages):
        import time
        import sys

        start_time = time.time()

        if first_call[0]:
            print(f"[INFO] ========== 这是首次推理调用 ==========")
            first_call[0] = False
        else:
            print(f"[INFO] ========== 这是第 N 次推理调用 ==========")

        print(f"[DEBUG] ========== 开始推理 ==========")
        print(f"[DEBUG] Python version: {sys.version}")
        print(f"[DEBUG] PyTorch version: {mindtorch.__version__}")
        print(f"[DEBUG] CUDA available: {mindtorch.cuda.is_available()}")
        if mindtorch.cuda.is_available():
            print(f"[DEBUG] CUDA device count: {mindtorch.cuda.device_count()}")
            print(f"[DEBUG] Current CUDA device: {mindtorch.cuda.current_device()}")
            print(f"[DEBUG] Device name: {mindtorch.cuda.get_device_name(0)}")
            print(
                f"[DEBUG] GPU Memory allocated: {mindtorch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            )
            print(
                f"[DEBUG] GPU Memory reserved: {mindtorch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
            )

        # 关键：检查并确保模型在 GPU 上
        model_device = next(model.parameters()).device
        print(f"[DEBUG] Model device: {model_device}")
        print(f"[DEBUG] Model dtype: {next(model.parameters()).dtype}")

        if str(model_device) == "cpu":
            print(f"[ERROR] 模型在 CPU 上！尝试移动到 GPU...")
            if mindtorch.cuda.is_available():
                move_start = time.time()
                model.cuda()
                move_time = time.time() - move_start
                print(
                    f"[DEBUG] Model device after cuda(): {next(model.parameters()).device}"
                )
                print(f"[DEBUG] 模型移动到 GPU 耗时: {move_time:.2f}s")
            else:
                print(f"[CRITICAL] CUDA 不可用！将在 CPU 上运行，速度会很慢！")
                print(f"[CRITICAL] 这可能是因为 ZeroGPU 资源紧张或超时")
        else:
            print(f"[INFO] 模型已在 GPU 上: {model_device}")

        messages = [messages]

        # 使用 processor 构造输入格式
        texts = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        print(f"[DEBUG] 模板构建完成，耗时: {time.time() - start_time:.2f}s")

        image_inputs, video_inputs = process_vision_info(messages)
        print(f"[DEBUG] 图像处理完成，耗时: {time.time() - start_time:.2f}s")

        # 检查图像输入大小
        if image_inputs:
            for idx, img in enumerate(image_inputs):
                if hasattr(img, "size"):
                    print(f"[DEBUG] Image {idx} size: {img.size}")
                elif isinstance(img, np.ndarray):
                    print(f"[DEBUG] Image {idx} shape: {img.shape}")

        print(f"[DEBUG] 开始 processor 编码输入...")
        processor_start = time.time()

        print(f"[DEBUG] 开始 processor 编码输入...")
        processor_start = time.time()
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        print(f"[DEBUG] Processor 编码完成，耗时: {time.time() - processor_start:.2f}s")

        # 确保输入在 GPU 上
        to_device_start = time.time()
        inputs = inputs.to("cuda" if mindtorch.cuda.is_available() else "cpu")
        print(f"[DEBUG] 输入移到设备耗时: {time.time() - to_device_start:.2f}s")
        print(f"[DEBUG] 输入准备完成，总耗时: {time.time() - start_time:.2f}s")
        print(f"[DEBUG] Input IDs shape: {inputs.input_ids.shape}")
        print(f"[DEBUG] Input device: {inputs.input_ids.device}")
        print(f"[DEBUG] Input sequence length: {inputs.input_ids.shape[1]}")

        # 生成
        gen_start = time.time()
        print(f"[DEBUG] ========== 开始生成 tokens ==========")

        # 关键优化：根据任务类型动态调整 max_new_tokens
        # OCR 任务通常不需要 8192 tokens，这会导致不必要的等待
        max_new_tokens = 2048  # 从 8192 降到 2048，大幅提速
        print(f"[DEBUG] max_new_tokens: {max_new_tokens}")

        # 添加进度回调
        token_count = [0]
        last_time = [gen_start]

        def progress_callback(input_ids, scores, **kwargs):
            token_count[0] += 1
            current_time = time.time()
            if token_count[0] % 10 == 0 or (current_time - last_time[0]) > 2.0:
                elapsed = current_time - gen_start
                tokens_per_sec = token_count[0] / elapsed if elapsed > 0 else 0
                print(
                    f"[DEBUG] 已生成 {token_count[0]} tokens, 速度: {tokens_per_sec:.2f} tokens/s, 耗时: {elapsed:.2f}s"
                )
                last_time[0] = current_time
            return False

        with mindtorch.no_grad():
            print(
                f"[DEBUG] 进入 mindtorch.no_grad() 上下文，耗时: {time.time() - start_time:.2f}s"
            )

            # 先做一次简单的前向传播测试
            print(f"[DEBUG] 测试前向传播...")
            forward_test_start = time.time()
            try:
                with mindtorch.cuda.amp.autocast(dtype=mindtorch.bfloat16):
                    test_outputs = model(**inputs, use_cache=False)
                print(
                    f"[DEBUG] 前向传播测试成功，耗时: {time.time() - forward_test_start:.2f}s"
                )
            except Exception as e:
                print(f"[WARNING] 前向传播测试失败: {e}")

            print(
                f"[DEBUG] 开始调用 model.generate()... (当前耗时: {time.time() - start_time:.2f}s)"
            )
            generate_call_start = time.time()

            try:
                # 关键：添加更激进的生成参数，强制早停
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0,
                )
                print(
                    f"[DEBUG] model.generate() 返回，耗时: {time.time() - generate_call_start:.2f}s"
                )
            except Exception as e:
                print(f"[ERROR] 生成失败: {e}")
                import traceback

                traceback.print_exc()
                raise

        print(f"[DEBUG] 退出 mindtorch.no_grad() 上下文")

        gen_time = time.time() - gen_start
        print(f"[DEBUG] ========== 生成完成 ==========")
        print(f"[DEBUG] 生成耗时: {gen_time:.2f}s")
        print(f"[DEBUG] Output shape: {generated_ids.shape}")

        # 解码输出
        if "input_ids" in inputs:
            input_ids = inputs.input_ids
        else:
            input_ids = inputs.inputs

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        actual_tokens = len(generated_ids_trimmed[0])
        print(f"[DEBUG] 实际生成 token 数: {actual_tokens}")
        print(
            f"[DEBUG] 每 token 耗时: {gen_time/actual_tokens if actual_tokens > 0 else 0:.3f}s"
        )

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        total_time = time.time() - start_time
        print(f"[DEBUG] ========== 全部完成 ==========")
        print(f"[DEBUG] 总耗时: {total_time:.2f}s")
        print(f"[DEBUG] 输出长度: {len(output_texts[0])} 字符")
        print(f"[DEBUG] 输出预览: {output_texts[0][:100]}...")
        output_texts[0] = clean_repeated_substrings(output_texts[0])
        return output_texts

    def create_predict_fn():

        def predict(_chatbot, task_history):
            nonlocal model, processor
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print("User: ", query)
            history_cp = copy.deepcopy(task_history)
            full_response = ""
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    # 判断是URL还是本地路径
                    img_path = q[0]
                    if img_path.startswith(("http://", "https://")):
                        content.append({"type": "image", "image": img_path})
                    else:
                        content.append(
                            {"type": "image", "image": f"{os.path.abspath(img_path)}"}
                        )
                else:
                    content.append({"type": "text", "text": q})
                    messages.append({"role": "user", "content": content})
                    messages.append(
                        {"role": "assistant", "content": [{"type": "text", "text": a}]}
                    )
                    content = []
            messages.pop()

            # 调用模型获取响应（已修改：不再传递 model 和 processor）
            response_list = call_local_model(messages)
            response = response_list[0] if response_list else ""

            _chatbot[-1] = (
                _parse_text(chat_query),
                _remove_image_special(_parse_text(response)),
            )
            full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print("HunyuanOCR: " + _parse_text(full_response))
            yield _chatbot

        return predict

    def create_regenerate_fn():

        def regenerate(_chatbot, task_history):
            nonlocal model, processor
            if not task_history:
                return _chatbot
            item = task_history[-1]
            if item[1] is None:
                return _chatbot
            task_history[-1] = (item[0], None)
            chatbot_item = _chatbot.pop(-1)
            if chatbot_item[0] is None:
                _chatbot[-1] = (_chatbot[-1][0], None)
            else:
                _chatbot.append((chatbot_item[0], None))
            # 使用外层的predict函数
            _chatbot_gen = predict(_chatbot, task_history)
            for _chatbot in _chatbot_gen:
                yield _chatbot

        return regenerate

    predict = create_predict_fn()
    regenerate = create_regenerate_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def download_url_image(url):
        """下载 URL 图片到本地临时文件"""
        try:
            # 使用 URL 的哈希值作为文件名，避免重复下载
            url_hash = hashlib.md5(url.encode()).hexdigest()
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"hyocr_demo_{url_hash}.png")

            # 如果文件已存在，直接返回
            if os.path.exists(temp_path):
                return temp_path

            # 下载图片
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(response.content)
            return temp_path
        except Exception as e:
            print(f"下载图片失败: {url}, 错误: {e}")
            return url  # 失败时返回原 URL

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, task_history):
        task_history.clear()
        _chatbot.clear()
        _gc()
        return []

    # 示例图片路径配置 - 请替换为实际图片路径
    EXAMPLE_IMAGES = {
        "spotting": "examples/spotting.jpg",
        "parsing": "examples/parsing.jpg",
        "ie": "examples/ie.jpg",
        "vqa": "examples/vqa.jpg",
        "translation": "examples/translation.jpg",
    }

    with gr.Blocks(
        css="""
        body {
            background: #f5f7fa;
        }
        .gradio-container {
            max-width: 100% !important;
            padding: 0 40px !important;
        }
        .header-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px 0;
            margin: -20px -40px 30px -40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            padding: 0 40px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .header-logo {
            height: 60px;
        }
        .header-text h1 {
            color: white;
            font-size: 32px;
            font-weight: bold;
            margin: 0 0 5px 0;
        }
        .header-text p {
            color: rgba(255,255,255,0.9);
            margin: 0;
            font-size: 14px;
        }
        .main-container {
            max-width: 1800px;
            margin: 0 auto;
        }
        .chatbot {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
            border-radius: 12px !important;
            border: 1px solid #e5e7eb !important;
            background: white !important;
        }
        .input-panel {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid #e5e7eb;
        }
        .input-box textarea {
            border: 2px solid #e5e7eb !important;
            border-radius: 8px !important;
            font-size: 14px !important;
        }
        .input-box textarea:focus {
            border-color: #667eea !important;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 500 !important;
            padding: 10px 24px !important;
            font-size: 14px !important;
        }
        .btn-primary:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        }
        .btn-secondary {
            background: white !important;
            border: 2px solid #667eea !important;
            color: #667eea !important;
            padding: 8px 20px !important;
            font-size: 14px !important;
        }
        .btn-secondary:hover {
            background: #f0f4ff !important;
        }
        .example-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        .example-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid #e5e7eb;
            transition: all 0.3s ease;
        }
        .example-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.15);
            border-color: #667eea;
        }
        .example-image-wrapper {
            width: 100%;
            height: 180px;
            overflow: hidden;
            background: #f5f7fa;
        }
        .example-image-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .example-btn {
            width: 100% !important;
            white-space: pre-wrap !important;
            text-align: left !important;
            padding: 16px !important;
            background: white !important;
            border: none !important;
            border-top: 1px solid #e5e7eb !important;
            color: #1f2937 !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
        }
        .example-btn:hover {
            background: #f9fafb !important;
            color: #667eea !important;
        }
        .feature-section {
            background: white;
            padding: 24px;
            border-radius: 12px;
            margin-top: 30px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid #e5e7eb;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e5e7eb;
        }
    """
    ) as demo:
        # 顶部导航栏
        gr.HTML(
            """
        <div class="header-section">
            <div class="header-content">
                <div class="header-text">
                    <h1>HunyuanOCR + MindNLP</h1>
                    <p>Powered by Tencent Hunyuan Team and MindSpore Team</p>
                </div>
            </div>
        </div>
        """
        )

        with gr.Column(elem_classes=["main-container"]):
            # 对话区域 - 全宽
            chatbot = gr.Chatbot(
                label="💬 对话窗口",
                height=600,
                bubble_full_width=False,
                layout="bubble",
                show_copy_button=True,
                elem_classes=["chatbot"],
            )

            # 输入控制面板 - 全宽
            with gr.Group(elem_classes=["input-panel"]):
                query = gr.Textbox(
                    lines=2,
                    label="💭 输入您的问题",
                    placeholder="请先上传图片，然后输入问题。例如：检测并识别图片中的文字，将文本坐标格式化输出。",
                    elem_classes=["input-box"],
                    show_label=False,
                )

                with gr.Row():
                    addfile_btn = gr.UploadButton(
                        "📁 上传图片",
                        file_types=["image"],
                        elem_classes=["btn-secondary"],
                    )
                    submit_btn = gr.Button(
                        "🚀 发送消息",
                        variant="primary",
                        elem_classes=["btn-primary"],
                        scale=3,
                    )
                    regen_btn = gr.Button("🔄 重新生成", elem_classes=["btn-secondary"])
                    empty_bin = gr.Button("🗑️ 清空对话", elem_classes=["btn-secondary"])

            # 示例区域 - 5列网格布局
            gr.HTML(
                '<div class="section-title">📚 快速体验示例 - 点击下方卡片快速加载</div>'
            )

            with gr.Row():
                # 示例1：spotting
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["example-card"]):
                        gr.Image("examples/spotting.jpg", height=180, elem_id="example-image-1")
                        example_1_btn = gr.Button(
                            "🔍 文字检测和识别", elem_classes=["example-btn"]
                        )

                # 示例2：parsing
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["example-card"]):
                        gr.Image("examples/parsing.jpg", height=180, elem_id="example-image-2")
                        example_2_btn = gr.Button(
                            "📋 文档解析", elem_classes=["example-btn"]
                        )

                # 示例3：ie
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["example-card"]):
                        gr.Image("examples/ie.jpg", height=180, elem_id="example-image-3")
                        example_3_btn = gr.Button(
                            "🎯 信息抽取", elem_classes=["example-btn"]
                        )

                # 示例4：VQA
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["example-card"]):
                        gr.Image("examples/vqa.jpg", height=180, elem_id="example-image-4")
                        example_4_btn = gr.Button(
                            "💬 视觉问答", elem_classes=["example-btn"]
                        )

                # 示例5：translation
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["example-card"]):
                        gr.Image("examples/translation.jpg", height=180, elem_id="example-image-5")
                        example_5_btn = gr.Button(
                            "🌐 图片翻译", elem_classes=["example-btn"]
                        )

        task_history = gr.State([])

        # 示例1：文档识别
        def load_example_1(history, task_hist):
            prompt = "检测并识别图片中的文字，将文本坐标格式化输出。"
            image_path = EXAMPLE_IMAGES["spotting"]
            # # 下载 URL 图片到本地
            # image_path = download_url_image(image_url)
            # 清空对话历史
            history = []
            task_hist = []
            history = history + [((image_path,), None)]
            task_hist = task_hist + [((image_path,), None)]
            return history, task_hist, prompt

        # 示例2：场景文字
        def load_example_2(history, task_hist):
            prompt = "提取文档图片中正文的所有信息用markdown 格式表示，其中页眉、页脚部分忽略，表格用html 格式表达，文档中公式用latex 格式表示，按照阅读顺序组织进行解析。"
            image_path = EXAMPLE_IMAGES["parsing"]
            # # 下载 URL 图片到本地
            # image_path = download_url_image(image_url)
            # 清空对话历史
            history = []
            task_hist = []
            history = history + [((image_path,), None)]
            task_hist = task_hist + [((image_path,), None)]
            return history, task_hist, prompt

        # 示例3：表格提取
        def load_example_3(history, task_hist):
            prompt = "提取图片中的：['单价', '上车时间','发票号码', '省前缀', '总金额', '发票代码', '下车时间', '里程数'] 的字段内容，并且按照JSON格式返回。"
            image_path = EXAMPLE_IMAGES["ie"]
            # # 下载 URL 图片到本地
            # image_path = download_url_image(image_url)
            # 清空对话历史
            history = []
            task_hist = []
            history = history + [((image_path,), None)]
            task_hist = task_hist + [((image_path,), None)]
            return history, task_hist, prompt

        # 示例4：手写体
        def load_example_4(history, task_hist):
            prompt = "What is the highest life expectancy at birth of male?"
            image_path = EXAMPLE_IMAGES["vqa"]
            # # 下载 URL 图片到本地
            # image_path = download_url_image(image_url)
            # 清空对话历史
            history = []
            task_hist = []
            history = history + [((image_path,), None)]
            task_hist = task_hist + [((image_path,), None)]
            return history, task_hist, prompt

        # 示例5：翻译
        def load_example_5(history, task_hist):
            prompt = "将图中文字翻译为中文。"
            image_path = EXAMPLE_IMAGES["translation"]
            # 下载 URL 图片到本地
            # image_path = download_url_image(image_url)
            # 清空对话历史
            history = []
            task_hist = []
            history = history + [((image_path,), None)]
            task_hist = task_hist + [((image_path,), None)]
            return history, task_hist, prompt

        # 绑定事件
        example_1_btn.click(
            load_example_1, [chatbot, task_history], [chatbot, task_history, query]
        )
        example_2_btn.click(
            load_example_2, [chatbot, task_history], [chatbot, task_history, query]
        )
        example_3_btn.click(
            load_example_3, [chatbot, task_history], [chatbot, task_history, query]
        )
        example_4_btn.click(
            load_example_4, [chatbot, task_history], [chatbot, task_history, query]
        )
        example_5_btn.click(
            load_example_5, [chatbot, task_history], [chatbot, task_history, query]
        )

        submit_btn.click(
            add_text, [chatbot, task_history, query], [chatbot, task_history]
        ).then(predict, [chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(
            reset_state, [chatbot, task_history], [chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )
        addfile_btn.upload(
            add_file,
            [chatbot, task_history, addfile_btn],
            [chatbot, task_history],
            show_progress=True,
        )

        # 功能说明区域
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    """
                <div class="feature-section">
                    <div class="section-title">✨ 核心功能</div>
                    <ul style="line-height: 2; color: #4b5563; font-size: 14px; margin: 0; padding-left: 20px;">
                        <li><strong>🎯 高精度文字检测识别</strong> - 支持多场景文字检测与识别</li>
                        <li><strong>📐 智能文档解析</strong> - 自动识别文档结构，支持多粒度文档解析</li>
                        <li><strong>📋 信息提取</strong> - 支持30+高频卡证票据识别和结构化输出</li>
                        <li><strong>✏️ 视觉问答</strong> - 支持以文本为中心的开放式问答</li>
                        <li><strong>🌍 跨语言翻译</strong> - 支持中英互译及14+语种译为中英文</li>
                    </ul>
                </div>
                """
                )

            with gr.Column(scale=1):
                gr.HTML(
                    """
                <div class="feature-section">
                    <div class="section-title">💡 使用建议</div>
                    <ul style="line-height: 2; color: #4b5563; font-size: 14px; margin: 0; padding-left: 20px;">
                        <li><strong>推理框架</strong> - 正式生产推荐使用VLLM，以获取更好的推理性能和精度</li>
                        <li><strong>拍摄角度</strong> - 确保图片清晰，光线充足，分辨率适中，避免严重倾斜、遮挡或反光，正面拍摄效果最佳</li>
                        <li><strong>文件大小</strong> - 建议单张图片不超过 10MB，支持 JPG/PNG 格式</li>
                        <li><strong>使用场景</strong> - 适用于文字检测识别、文档数字化、票据识别、信息提取、文字图片翻译等</li>
                        <li><strong>合规使用</strong> - 仅供学习研究，请遵守法律法规，尊重隐私权</li>
                    </ul>
                </div>
                """
                )

        # 底部版权信息
        gr.HTML(
            """
        <div style="text-align: center; color: #9ca3af; font-size: 13px; margin-top: 40px; padding: 20px; border-top: 1px solid #e5e7eb;">
            <p style="margin: 0;">© 2025 Tencent Hunyuan Team. All rights reserved.</p>
            <p style="margin: 5px 0 0 0;">本系统基于 HunyuanOCR 构建 | 仅供学习研究使用</p>
        </div>
        """
        )
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        # server_port=args.server_port,
        # server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)


if __name__ == "__main__":
    main()
