#LLMtest_evaluate8_RAG.py
import os
import re
import json
import numpy as np
import jieba
import jieba.analyse
from fuzzywuzzy import fuzz
from rouge import Rouge
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from openai import OpenAI
# 在原import区域追加
from RAGOrchestrator import TeachingRAGOrchestrator
from SegmentTeachingRAG import SegmentTeachingRAG
import logging, httpx
logging.getLogger("httpx").setLevel(logging.WARNING)   # 只看 WARN 及以上

# 初始化客户端
client = OpenAI(base_url="http://10.154.22.11:8000/v1", api_key="EMPTY")

# ================= 配置区域 =================
MODELS = {
    #"DeepSeek-R1-Distill-Qwen-1.5B": "/home/lhr/7Bmodel/DeepSeek-R1-Distill-Qwen-1.5B",
    #"DeepSeek-R1-Distill-Qwen-7B":"/home/lhr/7Bmodel/DeepSeek-R1-Distill-Qwen-7B",
    #"Qwen1.5": "/home/lhr/7Bmodel/Qwen1.5-4B-Chat",
    "Qwen3": "/home/lhr/7Bmodel/Qwen3-4B",
    #"Mistral": "/home/lhr/7Bmodel/Chinese-Mistral-7B-Instruct-v0.1",
    #"ChatGLM3": "/home/lhr/7Bmodel/chatglm3-6b",
    #"Baichuan2": "/home/lhr/7Bmodel/Baichuan2-7B-Chat",
    #"InternLM2-5-7B-Chat-1M": "/home/lhr/7Bmodel/internlm2_5-7b-chat-1m",
    #"MiniCPM3-4B": "/home/lhr/7Bmodel/MiniCPM3-4B",
}

# ================= 模板配置（保持原样）=================
MODEL_TEMPLATES = {
    # ================= DeepSeek 系列 =================
    "deepseek1.5b": {
        "template": """
        {% if not add_generation_prompt is defined %}
            {% set add_generation_prompt = false %}
        {% endif %}
        {% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}
        {%- for message in messages %}
            {%- if message['role'] == 'system' %}
                {% set ns.system_prompt = message['content'] %}
            {%- endif %}
        {%- endfor %}
        {{bos_token}}{{ns.system_prompt}}
        {%- for message in messages %}
            {%- if message['role'] == 'user' %}
                {%- set ns.is_tool = false -%}
                {{'### ' + message['content']}}
            {%- endif %}
            {%- if message['role'] == 'assistant' and message['content'] is none %}
                {%- set ns.is_tool = false -%}
                {%- for tool in message['tool_calls']%}
                    {%- if not ns.is_first %}
                        {{'### ' + tool['type'] + '\n' + tool['function']['name'] + '\n```json\n' + tool['function']['arguments'] + '\n```\n'}}
                        {%- set ns.is_first = true -%}
                    {%- else %}
                        {{'\n### ' + tool['type'] + '\n' + tool['function']['name'] + '\n```json\n' + tool['function']['arguments'] + '\n```\n'}}
                        {{'### '}}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {%- if message['role'] == 'assistant' and message['content'] is not none %}
                {%- if ns.is_tool %}
                    {{'### ' + message['content'] + '\n'}}
                    {%- set ns.is_tool = false -%}
                {%- else %}
                    {% set content = message['content'] %}
                    {% if '</think>' in content %}
                        {% set content = content.split('</think>')[-1] %}
                    {% endif %}
                    {{'### ' + content + '\n'}}
                {%- endif %}
            {%- endif %}
            {%- if message['role'] == 'tool' %}
                {%- set ns.is_tool = true -%}
                {%- if ns.is_output_first %}
                    {{'### ' + message['content'] + '\n'}}
                    {%- set ns.is_output_first = false %}
                {%- else %}
                    {{'\n### ' + message['content'] + '\n'}}
                {%- endif %}
            {%- endif %}
        {%- endfor -%}
        {% if ns.is_tool %}
            {{'### '}}
        {% endif %}
        {% if add_generation_prompt and not ns.is_tool %}
            {{'### <think>\n'}}
        {% endif %}
        """,
        "keywords": ["DeepSeek-R1-Distill-Qwen-1.5B"],
        "features": {
            "supports_tools": True,
            "requires_system_prompt": False,
            "special_tokens": {
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|end_of_text|>"
            }
        }
    },

    "deepseek7b": {
            "template": """
            {% if not add_generation_prompt is defined %}
                {% set add_generation_prompt = false %}
            {% endif %}
            {% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}
            {%- for message in messages %}
                {%- if message['role'] == 'system' %}
                    {% set ns.system_prompt = message['content'] %}
                {%- endif %}
            {%- endfor %}
            {{bos_token}}{{ns.system_prompt}}
            {%- for message in messages %}
                {%- if message['role'] == 'user' %}
                    {%- set ns.is_tool = false -%}
                    {{'### ' + message['content']}}
                {%- endif %}
                {%- if message['role'] == 'assistant' and message['content'] is none %}
                    {%- set ns.is_tool = false -%}
                    {%- for tool in message['tool_calls']%}
                        {%- if not ns.is_first %}
                            {{'### ' + tool['type'] + '\n' + tool['function']['name'] + '\n```json\n' + tool['function']['arguments'] + '\n```\n'}}
                            {%- set ns.is_first = true -%}
                        {%- else %}
                            {{'\n### ' + tool['type'] + '\n' + tool['function']['name'] + '\n```json\n' + tool['function']['arguments'] + '\n```\n'}}
                            {{'### '}}
                        {%- endif %}
                    {%- endfor %}
                {%- endif %}
                {%- if message['role'] == 'assistant' and message['content'] is not none %}
                    {%- if ns.is_tool %}
                        {{'### ' + message['content'] + '\n'}}
                        {%- set ns.is_tool = false -%}
                    {%- else %}
                        {% set content = message['content'] %}
                        {% if '</think>' in content %}
                            {% set content = content.split('</think>')[-1] %}
                        {% endif %}
                        {{'### ' + content + '\n'}}
                    {%- endif %}
                {%- endif %}
                {%- if message['role'] == 'tool' %}
                    {%- set ns.is_tool = true -%}
                    {%- if ns.is_output_first %}
                        {{'### ' + message['content'] + '\n'}}
                        {%- set ns.is_output_first = false %}
                    {%- else %}
                        {{'\n### ' + message['content'] + '\n'}}
                    {%- endif %}
                {%- endif %}
            {%- endfor -%}
            {% if ns.is_tool %}
                {{'### '}}
            {% endif %}
            {% if add_generation_prompt and not ns.is_tool %}
                {{'### <think>\n'}}
            {% endif %}
            """,
            "keywords": ["DeepSeek-R1-Distill-Qwen-7B"],
            "features": {
                "supports_tools": True,
                "requires_system_prompt": True,
                "special_tokens": {
                    "bos_token": "<|begin_of_text|>",
                    "eos_token": "<|end_of_text|>"
                }
            }
        },

    # ================= Qwen1.5 系列 =================
    "qwen1.5": {
        "template": "{% for message in messages %}"
                    "{% if loop.first and messages[0]['role'] != 'system' %}"
                    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
                    "{% endif %}"
                    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>assistant\n' }}"
                    "{% endif %}",
        "keywords": ["Qwen1.5-4B-Chat"],  # 明确限定1.5版本
        "features": {
            "supports_tools": False,
            "requires_system_prompt": False
        }
    },

    # ================= Qwen3 系列 =================
    "qwen3": {
        "template": "{%- if tools %}\n"
                    "    {{- '<|im_start|>system\\n' }}\n"
                    "    {%- if messages[0]['role'] == 'system' %}\n"
                    "        {{- messages[0]['content'] + '\\n\\n' }}\n"
                    "    {%- endif %}\n"
                    "    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n"
                    "    {%- for tool in tools %}\n"
                    "        {{- \"\\n\" }}\n"
                    "        {{- tool | tojson }}\n"
                    "    {%- endfor %}\n"
                    "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n"
                    "{%- else %}\n"
                    "    {%- if messages[0]['role'] == 'system' %}\n"
                    "        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n"
                    "    {%- endif %}\n"
                    "{%- endif %}\n"
                    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
                    "{%- for message in messages[::-1] %}\n"
                    "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
                    "    {%- if ns.multi_step_tool and message['role'] == \"user\" and message['content'] is string and not(message['content'].startswith('<tool_response>') and message['content'].endswith('</tool_response>')) %}\n"
                    "        {%- set ns.multi_step_tool = false %}\n"
                    "        {%- set ns.last_query_index = index %}\n"
                    "    {%- endif %}\n"
                    "{%- endfor %}\n"
                    "{%- for message in messages %}\n"
                    "    {%- if message['content'] is string %}\n"
                    "        {%- set content = message['content'] %}\n"
                    "    {%- else %}\n"
                    "        {%- set content = '' %}\n"
                    "    {%- endif %}\n"
                    "    {%- if (message['role'] == \"user\") or (message['role'] == \"system\" and not loop.first) %}\n"
                    "        {{- '<|im_start|>' + message['role'] + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
                    "    {%- elif message['role'] == \"assistant\" %}\n"
                    "        {%- set reasoning_content = '' %}\n"
                    "        {%- if message.get('reasoning_content') is string %}\n"
                    "            {%- set reasoning_content = message['reasoning_content'] %}\n"
                    "        {%- else %}\n"
                    "            {%- if '</think>' in content %}\n"
                    "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
                    "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
                    "            {%- endif %}\n"
                    "        {%- endif %}\n"
                    "        {%- if loop.index0 > ns.last_query_index %}\n"
                    "            {%- if loop.last or (not loop.last and reasoning_content) %}\n"
                    "                {{- '<|im_start|>' + message['role'] + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
                    "            {%- else %}\n"
                    "                {{- '<|im_start|>' + message['role'] + '\\n' + content }}\n"
                    "            {%- endif %}\n"
                    "        {%- else %}\n"
                    "            {{- '<|im_start|>' + message['role'] + '\\n' + content }}\n"
                    "        {%- endif %}\n"
                    "        {%- if message.get('tool_calls') %}\n"
                    "            {%- for tool_call in message['tool_calls'] %}\n"
                    "                {%- if (loop.first and content) or (not loop.first) %}\n"
                    "                    {{- '\\n' }}\n"
                    "                {%- endif %}\n"
                    "                {%- if tool_call.get('function') %}\n"
                    "                    {%- set tool_call = tool_call['function'] %}\n"
                    "                {%- endif %}\n"
                    "                {{- '<tool_call>\\n{\"name\": \"' }}\n"
                    "                {{- tool_call['name'] }}\n"
                    "                {{- '\", \"arguments\": ' }}\n"
                    "                {%- if tool_call['arguments'] is string %}\n"
                    "                    {{- tool_call['arguments'] }}\n"
                    "                {%- else %}\n"
                    "                    {{- tool_call['arguments'] | tojson }}\n"
                    "                {%- endif %}\n"
                    "                {{- '}\\n</tool_call>' }}\n"
                    "            {%- endfor %}\n"
                    "        {%- endif %}\n"
                    "        {{- '<|im_end|>\\n' }}\n"
                    "    {%- elif message['role'] == \"tool\" %}\n"
                    "        {%- if loop.first or (messages[loop.index0 - 1]['role'] != \"tool\") %}\n"
                    "            {{- '<|im_start|>user' }}\n"
                    "        {%- endif %}\n"
                    "        {{- '\\n<tool_response>\\n' }}\n"
                    "        {{- content }}\n"
                    "        {{- '\\n</tool_response>' }}\n"
                    "        {%- if loop.last or (messages[loop.index0 + 1]['role'] != \"tool\") %}\n"
                    "            {{- '<|im_end|>\\n' }}\n"
                    "        {%- endif %}\n"
                    "    {%- endif %}\n"
                    "{%- endfor %}\n"
                    "{%- if add_generation_prompt %}\n"
                    "    {{- '<|im_start|>assistant\\n' }}\n"
                    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
                    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
                    "    {%- endif %}\n"
                    "{%- endif %}",
        "keywords": ["Qwen3-4B"],  # 明确限定3.x版本
        "features": {
            "supports_tools": True,
            "requires_system_prompt": False
        }
    },

    # ================= Mistral 系列 =================
    "mistral": {
        "template": "{{ bos_token }}"
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
                    "{% else %}"
                    "{{ message['content'] + eos_token }}"
                    "{% endif %}"
                    "{% endfor %}",
        "keywords": ["mistral"],
        "features": {
            "supports_tools": False,
            "requires_system_prompt": True
        }
    },

    # ================= ChatGLM 系列 =================
    "chatglm": {
        "template": "{% for message in messages %}"
                    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                    "{% endfor %}",
        "keywords": ["chatglm"],
        "features": {
            "supports_tools": False,
            "requires_system_prompt": True
        }
    },

    # ================= Baichuan 系列 =================
    "baichuan": {
        "template": "{{ bos_token }}"
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ '<reserved_106>' + message['content'] + '<reserved_107>' }}"
                    "{% else %}"
                    "{{ message['content'] + eos_token }}"
                    "{% endif %}"
                    "{% endfor %}",
        "keywords": ["baichuan"],
        "features": {
            "supports_tools": False,
            "requires_system_prompt": True
        }
    },

    # ================= InternLM2 系列 =================
    "internlm": {
        "template": "{% for message in messages %}"
                    "<|im_start|>{{message['role']}}\n{{message['content']}}<|im_end|>\n"
                    "{% endfor %}",
        "keywords": ["internlm"],
        "features": {
            "supports_tools": False,
            "requires_system_prompt": True
        }
    },

    # ================= MiniCPM 系列 =================
    "minicpm": {
        "template": "{% for message in messages %}"
                    "<|im_start|>{{message['role']}}\n{{message['content']}}<|im_end|>\n"
                    "{% endfor %}",
        "keywords": ["minicpm"],
        "features": {
            "supports_tools": False,
            "requires_system_prompt": True
        }
    }
}


def detect_model_type(model_path: str) -> str:
    """
    改进后的模型类型检测函数，优先精确匹配特定版本（如Qwen3/Qwen1.5）
    参数:
        model_path: 模型路径（不区分大小写）
    返回:
        匹配的模板名称（如 "qwen3", "deepseek"）
    """

    # 精确匹配优先（避免Qwen3和Qwen1.5混淆）
    if "DeepSeek-R1-Distill-Qwen-7B"  in model_path:
        return "deepseek7b"
    if "DeepSeek-R1-Distill-Qwen-1.5B" in model_path:
        return "deepseek1.5b"
    if "Qwen3-4B" in model_path:
        return "qwen3"
    if "Qwen1.5-4B-Chat" in model_path:
        return "qwen1.5"


    # 通用关键词匹配
    for model_type, config in MODEL_TEMPLATES.items():
        if any(kw in model_path for kw in config["keywords"]):
            return model_type

    return "mistral"  # 默认退回

# def detect_model_type(model_path):
#     model_path = model_path.lower()
#     for model_type, config in MODEL_TEMPLATES.items():
#         if any(kw in model_path for kw in config["keywords"]):
#             return model_type
#     return "mistral"


# ================= 评估模块 =================
class LessonPlanEvaluator:
    def __init__(self, gt_json_path: str):
        with open(gt_json_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        self.rouge = Rouge()
        jieba.initialize()

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """增强版的环节解析（支持更多格式变体）"""
        sections = {}
        current_section = None
        # 支持多种标记格式：【环节名称】、[环节名称]、环节名称：
        section_pattern = re.compile(r'^([【\[《])(.+?)([】\]》])$|^(.*?:)$')

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 尝试匹配环节名称
            match = section_pattern.match(line)
            if match:
                # 提取环节名称（兼容【A】或 A: 等形式）
                current_section = match.group(2) if match.group(2) else match.group(4).rstrip(':')
                sections[current_section] = ""
            elif current_section:
                sections[current_section] += line + "\n"

        return {k: v.strip() for k, v in sections.items() if k and v}

    def evaluate(self, pred_text: str, filename: str) -> Dict[str, float]:
        """执行完整评估流程"""

        pred_sections = self._parse_sections(pred_text)
        gt_sections = self.gt_data.get(filename, {})

        if not gt_sections:
            return {"error": "未找到对应的标准答案"}
        # 计算各项指标
        metrics = {
            "内容完整性": self._content_completeness(pred_sections, gt_sections),
            "顺序合理性": self._order_consistency(pred_sections, gt_sections),
            "环节边界准确率": self._boundary_accuracy(pred_text, gt_sections),
            "环节名称规范性": self._section_name_soft(pred_sections, gt_sections)
        }
        # 计算综合评分（加权平均）
        weights = {
            "内容完整性": 0.4,
            "顺序合理性": 0.3,
            "环节边界准确率": 0.2,
            "环节名称规范性": 0.1
        }
        metrics["综合评分"] = sum(metrics[k] * weights[k] for k in metrics)
        return {k: round(v, 4) for k, v in metrics.items()}

    # 内容完整性
    # def _content_completeness(self, pred: Dict[str, str], gt: Dict[str, str]) -> float:
    #     scores = []
    #     pred_contents = list(pred.values())
    #     for gt_content in gt.values():
    #         best_score = max([
    #             self.rouge.get_scores(pred_content, gt_content)[0]['rouge-l']['f']
    #             for pred_content in pred_contents
    #         ], default=0.0)
    #         scores.append(best_score)
    #     return np.mean(scores)

    def _content_completeness(self, pred: Dict[str, str], gt: Dict[str, str]) -> float:
        scores = []
        pred_contents = [v for v in pred.values() if v.strip()]  # 先过滤空串
        for gt_content in gt.values():
            if not gt_content.strip():
                scores.append(0.0)
                continue
            try:
                best = max([
                    self.rouge.get_scores(p, gt_content)[0]['rouge-l']['f']
                    for p in pred_contents
                ], default=0.0)
            except ValueError:  # ← 捕获 rouge 的空串异常
                best = 0.0
            scores.append(best)
        return float(np.mean(scores))

    # 顺序合理性
    def _order_consistency(self, pred: Dict[str, str], gt: Dict[str, str]) -> float:
        pred_texts = [text for text in pred.values() if text.strip()]
        gt_texts = [text for text in gt.values() if text.strip()]

        if not pred_texts or not gt_texts:
            return 0.0

        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, min_df=1)
        try:
            pred_matrix = vectorizer.fit_transform(pred_texts)
            gt_matrix = vectorizer.transform(gt_texts)
            sim_matrix = cosine_similarity(pred_matrix, gt_matrix)
            matched_indices = np.argmax(sim_matrix, axis=1)
            correct = sum(1 for i in range(len(matched_indices) - 1)
                          if matched_indices[i] < matched_indices[i + 1])
            return correct / (len(matched_indices) - 1) if len(matched_indices) > 1 else 0.0
        except:
            return 0.0

    def _boundary_accuracy(self, pred_text: str, gt_sections: Dict[str, str]) -> float:
        """
        改进后的环节边界准确率评估（不依赖start_pos，直接比较边界位置）
        参数:
            pred_text: 模型输出的教案文本
            gt_sections: 标准答案字典（{"环节名称": "内容"}）
        返回:
            边界准确率分数 (0~1之间)
        """
        # 1. 检测预测文本的所有边界位置
        pred_boundaries = self._detect_boundaries(pred_text)

        # 2. 生成标准答案的完整文本并检测边界
        gt_text = "\n".join([f"【{k}】\n{v}" for k, v in gt_sections.items()])
        gt_boundaries = self._detect_boundaries(gt_text)

        # 3. 如果没有预测到任何边界
        if not pred_boundaries:
            return 0.0

        # 4. 计算每个预测边界的匹配分数（50字以内渐进计分）
        total_score = 0.0
        for pred_pos in pred_boundaries:
            best_match_score = 0.0
            for gt_pos in gt_boundaries:
                distance = abs(pred_pos - gt_pos)
                if distance <= 50:  # 50字以内的容差范围
                    score = max(0, 1 - distance / 50)
                    if score > best_match_score:
                        best_match_score = score
            total_score += best_match_score

        # 5. 计算准确率（平均每个预测边界的匹配分数）
        boundary_accuracy = total_score / len(pred_boundaries)
        return round(boundary_accuracy, 4)

    def _detect_boundaries(self, text: str) -> List[int]:
        """
        检测文本中的所有环节边界位置（返回字符索引）
        支持多种标记格式：【环节名称】、[环节名称]、"环节名称"、环节名称：
        """
        boundary_patterns = [
            r'[【\[](.+?)[】\]]',  # 匹配【】或[]
            r'"(.+?)"',  # 匹配双引号
            r'(.+?)：',  # 匹配中文冒号
            r'^([\u4e00-\u9fa5]{2,}[\s]*)$'  # 匹配独立行中文（至少2字）
        ]
        boundaries = set()
        for pattern in boundary_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                boundaries.add(match.start())
        return sorted(boundaries)

    # 环节名称规范性
    def _section_name_soft(self, pred_sections: Dict[str, str], gt_sections: Dict[str, str]) -> float:
        standard_names = list(gt_sections.keys())
        matched_scores = [max(fuzz.partial_ratio(p, s) / 100 for s in standard_names)
                          for p in pred_sections.keys()]
        ratio = min(len(pred_sections) / len(gt_sections), 1.0)
        return ratio * np.mean(matched_scores) if matched_scores else 0


# ================= 批处理函数 =================
###############################################################################topk循环函数专用
def batch_process_with_eval(
    input_folder: str,
    output_folder: str,
    gt_json_path: str,
    rag_orchestrator: TeachingRAGOrchestrator = None   # ← 写这里
):

    if rag_orchestrator is None:
        rag_orchestrator = TeachingRAGOrchestrator(
            encoder_url="http://10.154.22.11:9000",
            reranker_url="http://10.154.22.11:9001"
        )
####################################################################topk循环函数专用

#############################################################################################原始代码
#def batch_process_with_eval(input_folder: str, output_folder: str, gt_json_path: str,):
    # 新增RAG初始化 =================================
    # rag_orchestrator = TeachingRAGOrchestrator(
    #     encoder_url="http://10.154.22.11:9000",  # 根据实际修改
    #     reranker_url="http://10.154.22.11:9001",
    #     knowledge_base_path=r"D:\研究生\项目组\科研\小模型+RAG暑期实验\RAG知识库_3", # 您的知识库路径
    #     timeout=None  # ← 改为 None
    # )
#############################################################################################原始代码

    evaluator = LessonPlanEvaluator(gt_json_path)

    for model_name, model_path in MODELS.items():
        print(f"\n=== 开始测试模型: {model_name} ===")
        model_output_dir = os.path.join(output_folder, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        all_metrics = {}
        file_count = 0
        success_count = 0

        for filename in sorted(os.listdir(input_folder)):
            if not filename.endswith('.txt'):
                continue

            file_count += 1
            print(f"\n处理文件 {file_count}: {filename}")

            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # ============== 新增RAG调用（含错误处理）==============
            try:
                # -------------------------------------hybrid--------------------------------------------#
                #rag_results = rag_orchestrator.process(content, mode="hybrid")  # 默认用hybrid模式
                #rag_context = "\n".join([f"- {node[:200]}..." for node in rag_results[:3]]) # 取Top3结果
                #rag_context = "\n".join([node for node in rag_results[:3]])
                # rag_context = "\n".join([node[:node.find('。') + 1] or node
                #                          for node in rag_results[:3]])
                #--------------------------------------segment-------------------------------------------#
                rag_context = rag_orchestrator.process(content, mode="segment")

            except Exception as e:
                print(f"RAG检索失败: {str(e)}")
                rag_context = ""  # 失败时置空
            # ===================================================

            messages = [
                {
                    "role": "system",
                    "content": "你是一个资深教学设计师，请严格完成以下任务：\n"
                               "1. 识别教案中所有教学环节，并**按原始正确顺序**划分\n"
                               "2. 每个环节必须包含：【环节名称】和对应的完整教学内容\n"
                               "3. 环节名称需符合标准教学术语（如导入、讲解、练习、总结）\n"
                               "4. 保持内容原封不动，仅添加环节标记\n\n"
                               "**顺序复原规则**：\n"
                               "- 若教案环节顺序被打乱（如练习出现在导入前），需按教学逻辑复原\n"
                               "- 常见教学流程：导入 → 讲解 → 练习 → 总结 → 作业\n\n"
                               "输出格式要求：\n"
                               "【环节名称】\n"
                               "该环节的完整内容（包括所有文字、标点等原始内容）\n\n"
                               "示例：\n"
                               "【课堂导入】\n"
                               "同学们，今天我们学习...（此处为完整的导入部分内容）\n"
                               "【知识讲解】\n"
                               "分数由分子和分母组成..."
                },
                # ============== 插入RAG检索结果 ================
                {
                    "role": "assistant",
                    "content": "【相关教学知识参考】\n" + rag_context if rag_context else "未找到相关教学参考"
                },
                # =============================================
                {
                    "role": "user",
                    "content": f"请分析以下教案并划分教学环节（需复原正确顺序）：\n{content}"
                }
            ]

            try:
                response = client.chat.completions.create(
                    model=model_path,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    extra_body={"chat_template": MODEL_TEMPLATES[detect_model_type(model_path)]["template"]}
                )
                output = response.choices[0].message.content
                output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
                output = re.sub(r'^.*?</think>\n', '', output, flags=re.DOTALL)
                print(f"文件 {filename} 处理成功")
                success_count += 1
            except Exception as e:
                output = f"处理失败: {str(e)}"
                print(f"文件 {filename} 处理失败: {str(e)}")

            output_path = os.path.join(model_output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)

            if not output.startswith("处理失败"):
                metrics = evaluator.evaluate(output, filename)
                all_metrics[filename] = metrics
                print(f"评估结果: {metrics}")

        # 计算汇总统计
        summary = {
            "total_files": file_count,
            "success_files": success_count,
            "success_rate": success_count / file_count if file_count > 0 else 0
        }

        if all_metrics:
            avg_scores = {k: np.mean([m[k] for m in all_metrics.values()])
                          for k in all_metrics[next(iter(all_metrics))]}
            summary["average_metrics"] = avg_scores

            print(f"\n模型 {model_name} 评估汇总：")
            print(f"总文件数: {file_count}, 成功处理: {success_count}, 成功率: {summary['success_rate']:.2%}")
            print("平均指标:")
            for metric, score in avg_scores.items():
                print(f"{metric}: {score:.4f}")
        else:
            summary["average_metrics"] = None
            print("\n没有成功处理的文件，无法计算评估指标")

        metrics_path = os.path.join(output_folder, f"{model_name}_eval.json")
        result_data = {
            "detailed_metrics": all_metrics,
            "summary": summary
        }

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n评估结果已保存到: {metrics_path}")


# ================= 主程序 =================
if __name__ == "__main__":
    batch_process_with_eval(
        input_folder=r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试集\测试集2_删去环节名称并打乱顺序",
        output_folder=r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\测试结果\Segment_RAG测试结果",
        gt_json_path=r"D:\研究生\项目组\科研\小模型+RAG暑期实验\50个含教学环节名称的教案\txt教案\教学环节txt_json\teaching_plan.json"
    )
