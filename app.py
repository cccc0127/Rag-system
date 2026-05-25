import importlib.util
import json
import os
import shutil
import time

import gradio as gr
import psutil
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config
from main import rebuild_index
from rag import generate_answer


USER_ICON_PATH = "icon/user.png"
BOT_ICON_PATH = "icon/bot.png"

embedding_model = SentenceTransformer(config.EMBEDDING_MODEL,device=config.EMBEDDING_DEVICE)

def load_llm_model(model_name):
    model_path = str(config.MODEL_PATHS.get(model_name, config.LLM_MODEL_PATH))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if importlib.util.find_spec("accelerate") is not None:
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    return tokenizer, model


current_model_name = config.DEFAULT_LLM_MODEL
tokenizer, model = load_llm_model(current_model_name)


def upload_files(files, chatbot):
    chatbot = chatbot or []
    if not isinstance(files, list):
        files = [files]

    saved_files = []
    failed_files = []

    for file in files:
        try:
            original_filename = file.orig_name if hasattr(file, "orig_name") else os.path.basename(file.name)
            dest_path = os.path.join(config.REFERENCE_FOLDER, original_filename)
            shutil.move(file.name, dest_path)
            saved_files.append(original_filename)
        except Exception as exc:
            failed_files.append(original_filename)
            print(f"上传失败: {original_filename}, 错误: {exc}")

    if saved_files:
        print("至少一个文件上传成功，开始重建索引...")
        index_message = rebuild_index()
    else:
        index_message = "没有文件上传成功，索引未更新。"

    message = f"上传成功 {len(saved_files)} 个文件: {', '.join(saved_files)}"
    if failed_files:
        message += f"\n上传失败 {len(failed_files)} 个文件: {', '.join(failed_files)}"
    message += f"\n{index_message}"
    print(message)

    chatbot.append({"role": "user", "content": "上传文件"})
    chatbot.append({"role": "assistant", "content": message})
    return chatbot


def chat_with_rag(question, chatbot, max_tokens, temperature, top_p, show_debug, topk_retrieval, dist_threshold):
    chatbot = chatbot or []
    chatbot.append({"role": "user", "content": question})
    chatbot.append({"role": "assistant", "content": ""})
    start_time = time.time()

    current_output = generate_answer(question)
    bot_response = ""
    debug_info = ""

    if "</think>" in current_output:
        bot_response, debug_info = current_output.split("</think>", 1)
        bot_response = bot_response.strip()
        debug_info = debug_info.strip()
    else:
        bot_response = current_output.strip()

    elapsed_time = time.time() - start_time
    elapsed_str = f"点击查看推理过程，耗时 {elapsed_time:.2f} 秒"

    if show_debug and debug_info:
        bot_response = (
            f"<details>"
            f"<summary style='color:#888;font-size:12px;'>{elapsed_str}</summary>"
            f"<div style='color:#ccc;background:#f5f5f5;padding:10px;border-radius:5px;'>"
            f"{debug_info}"
            f"</div></details>\n\n"
            f"{bot_response}"
        )

    chatbot[-1] = {"role": "assistant", "content": bot_response}
    return "", chatbot


def switch_model(new_model):
    global tokenizer, model, current_model_name
    tokenizer, model = load_llm_model(new_model)
    current_model_name = new_model
    return f"已切换到 {new_model} 模型"


def system_diagnosis():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {
        "CPU 使用率": f"{cpu_usage}%",
        "内存使用率": f"{ram_usage}%",
        "GPU 占用": f"{gpu_usage:.2f}GB",
    }


def export_chat_history(chatbot):
    chatbot = chatbot or []
    file_path = "chat_history.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chatbot, f, ensure_ascii=False, indent=2)
    return file_path


def import_chat_history(file):
    if file is None:
        return []

    with open(file.name, "r", encoding="utf-8") as f:
        history = json.load(f)

    return history


def create_gradio_interface():
    with gr.Blocks(title="DeepSeek RAG System 2.0") as interface:
        gr.Markdown("# DeepSeek RAG 知识管理系统")

        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": "您好，我是 Theodore，您的智能助手。"}],
            height=680,
            avatar_images=(USER_ICON_PATH, BOT_ICON_PATH),
        )
        msg_input = gr.Textbox(placeholder="输入您的问题...", lines=3)

        with gr.Row():
            submit_btn = gr.Button("发送", variant="primary")
            os.environ["GRADIO_MAX_FILE_SIZE"] = "100mb"
            upload_btn = gr.UploadButton(
                "上传文档",
                file_types=[".pdf", ".docx", ".txt", ".md", ".pptx", ".xlsx"],
                file_count="multiple",
            )
            clear_btn = gr.Button("清空对话")

        with gr.Accordion("系统监控", open=False):
            gr.Markdown("### 实时系统指标")
            diagnose_btn = gr.Button("刷新状态")
            status_panel = gr.JSON(label="系统状态", value={"状态": "正在获取..."})

        interface.load(system_diagnosis, inputs=None, outputs=status_panel)

        with gr.Accordion("模型管理", open=False):
            model_selector = gr.Dropdown(
                label="选择模型",
                choices=list(config.MODEL_PATHS.keys()),
                value=current_model_name,
            )
            model_status = gr.Textbox(label="模型状态", interactive=False, value="正在初始化模型...")

        interface.load(lambda: switch_model(current_model_name), inputs=None, outputs=model_status)

        with gr.Accordion("对话历史", open=False):
            export_btn = gr.Button("导出历史")
            import_btn = gr.UploadButton("导入历史", file_types=[".json"])
            export_btn.click(export_chat_history, inputs=chatbot, outputs=gr.File())
            import_btn.upload(import_chat_history, inputs=import_btn, outputs=chatbot)

        with gr.Accordion("生成参数", open=False):
            max_tokens = gr.Slider(128, 4096, value=512, label="生成长度限制")
            temperature = gr.Slider(0.1, 1.0, value=0.7, label="温度")
            top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
            topk_retrieval = gr.Slider(1, 10, value=3, step=1, label="检索文档数 top_k")
            dist_threshold = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="检索距离阈值")
            show_debug = gr.Checkbox(label="显示推理过程", value=True)

        msg_input.submit(
            chat_with_rag,
            inputs=[msg_input, chatbot, max_tokens, temperature, top_p, show_debug, topk_retrieval, dist_threshold],
            outputs=[msg_input, chatbot],
        )
        submit_btn.click(
            chat_with_rag,
            inputs=[msg_input, chatbot, max_tokens, temperature, top_p, show_debug, topk_retrieval, dist_threshold],
            outputs=[msg_input, chatbot],
        )
        clear_btn.click(
            lambda: [{"role": "assistant", "content": "对话已清空。"}],
            outputs=chatbot,
        )

        upload_btn.upload(upload_files, inputs=[upload_btn, chatbot], outputs=[chatbot])
        diagnose_btn.click(system_diagnosis, outputs=status_panel)
        model_selector.change(switch_model, inputs=model_selector, outputs=model_status)

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="127.0.0.1", server_port=7860, share=False)
