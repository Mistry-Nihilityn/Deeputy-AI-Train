import json
from pprint import pprint

from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
import logging
from typing import Dict, List, Any, Optional
import re
from zai import ZhipuAiClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
client = ZhipuAiClient(api_key="c08862d32c657c00c90517a3d2d4764a.SVhS0RqKR1WIC1Uu")


# 初始化分类器
class ClassifierSingleton:
    _instance = None
    _classifier = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassifierSingleton, cls).__new__(cls)
            # 初始化分类器
            logger.info("正在加载零样本分类模型...")
            try:
                cls._classifier = pipeline(
                    'zero-shot-classification',
                    'damo/nlp_structbert_zero-shot-classification_chinese-base'
                )
                logger.info("模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
        return cls._instance

    def get_classifier(self):
        return self._classifier


try:
    classifier_singleton = ClassifierSingleton()
    logger.info("Flask应用启动，模型已初始化")
except Exception as e:
    logger.error(f"启动时初始化失败: {e}")


# 工具函数
def clean_content(content: str) -> str:
    """清理话术内容中的变量占位符"""
    # 移除 {xxx} 格式的变量
    cleaned = re.sub(r'\{.*?\}', '', content)
    # 移除多余空格和换行
    cleaned = cleaned.strip().replace('\n', ' ')
    # 合并多个空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def extract_candidate_texts(script_item: Dict) -> List[str]:
    """从话术项中提取用于分类的文本"""
    texts = []

    # 从多个字段提取文本
    if script_item.get('content'):
        texts.append(clean_content(script_item['content']))
    if script_item.get('summary'):
        texts.append(script_item['summary'])
    if script_item.get('name'):
        texts.append(script_item['name'])
    if script_item.get('tag'):
        texts.append(script_item['tag'])

    # 合并所有文本
    return texts


def filter_enabled_scripts(scripts: List[Dict]) -> List[Dict]:
    """过滤启用的话术"""
    enabled_scripts = []
    for script in scripts:
        # enabled为null或true的话术都认为是启用的
        if script.get('enabled') is not False:
            enabled_scripts.append(script)
    return enabled_scripts


def prepare_dialogue_context(dialogue_history: List[Dict]) -> str:
    """准备对话上下文文本"""
    if not dialogue_history:
        return ""

    # 取最后几条对话作为上下文
    recent_dialogues = dialogue_history[-4:]  # 取最近4条

    # 构建上下文文本
    context_parts = []
    for i, turn in enumerate(recent_dialogues):
        role_map = {
            'customer': '客户',
            'user': '客服',
            'system': '系统'
        }
        role = role_map.get(turn['role'], turn['role'])
        content = turn['content']
        context_parts.append(f"{role}: {content}")

    return " ".join(context_parts)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'service': 'zero-shot-classification-api'
    })


@app.route('/classify', methods=['POST'])
def classify():
    """
    零样本分类接口
    输入格式:
    {
        "nodeScript": [...],
        "feedbackScript": [...],
        "commonScript": [...],
        "dialogue": [
            {"role": "customer", "time": "...", "content": "..."},
            ...
        ]
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400

        # 获取对话历史
        dialogue_history = data.get('dialogue', None)
        if not dialogue_history:
            return jsonify({
                'success': False,
                'error': '对话历史不能为空'
            }), 400

        # 获取所有话术
        all_scripts = []
        script_sources = []
        script_ids = []

        # 处理nodeScript
        node_scripts = data.get('nodeScript', [])
        enabled_node_scripts = filter_enabled_scripts(node_scripts)
        all_scripts.extend(enabled_node_scripts)
        script_sources.extend(['node'] * len(enabled_node_scripts))
        script_ids.extend([s["id"] for s in enabled_node_scripts])

        # 处理feedbackScript
        feedback_scripts = data.get('feedbackScript', [])
        enabled_feedback_scripts = filter_enabled_scripts(feedback_scripts)
        all_scripts.extend(enabled_feedback_scripts)
        script_sources.extend(['feedback'] * len(enabled_feedback_scripts))
        script_ids.extend([s["id"] for s in enabled_feedback_scripts])

        # 处理commonScript
        common_scripts = data.get('commonScript', [])
        enabled_common_scripts = filter_enabled_scripts(common_scripts)
        all_scripts.extend(enabled_common_scripts)
        script_sources.extend(['common'] * len(enabled_common_scripts))
        script_ids.extend([s["id"] for s in enabled_common_scripts])

        if not all_scripts:
            return jsonify({
                'success': False,
                'error': '没有可用的话术'
            }), 400

        # 准备候选标签文本
        candidate_labels = []
        for script in all_scripts:
            # 提取话术的主要文本内容
            script_texts = extract_candidate_texts(script)
            if script_texts:
                # 使用content作为主要标签文本
                candidate_labels.append(script_texts[0])
            else:
                candidate_labels.append("")

        # 准备输入文本
        last_customer_message = None
        for turn in reversed(dialogue_history):
            if turn.get('role') == 'customer':
                last_customer_message = turn.get('content', '')
                break

        if not last_customer_message:
            last_customer_message = dialogue_history[-1].get('content', '') if dialogue_history else ""

        # 添加对话上下文
        context = prepare_dialogue_context(dialogue_history)
        input_text = f"{context} 客户说: {last_customer_message}; 请判断客服应该如何回复"

        logger.info(f"输入文本: {input_text}")
        logger.info(f"候选标签数量: {len(candidate_labels)}")

        classifier = classifier_singleton.get_classifier()

        if not classifier:
            return jsonify({
                'success': False,
                'error': '分类器未初始化'
            }), 500

        # 执行分类
        result = classifier(input_text, candidate_labels=candidate_labels, multi_label=True)

        return jsonify({
            'success': True,
            'message': "AI recommendation succeeded.",
            'data': {
                'ids': script_ids,
                'scores': [float(score) for score in result['scores']]
            }
        })

    except Exception as e:
        logger.error(f"分类处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/navigate', methods=['POST'])
def navigate():
    """
    零样本分类接口
    输入格式:
    {
        "nodeScript": [...],
        "feedbackScript": [...],
        "commonScript": [...],
        "dialogue": [
            {"role": "customer", "time": "...", "content": "..."},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400

        dialogue_history = data.get('dialogue', None)
        if not dialogue_history:
            return jsonify({
                'success': False,
                'error': '对话历史不能为空'
            }), 400

        out_edges = data.get('outEdges', [])
        out_edges_ids = [s["edgeId"] for s in out_edges]

        if len(out_edges_ids) == 0:
            return jsonify({
                'success': False,
                'error': '没有可用的话术'
            }), 400

        candidate_labels = []
        for edge in out_edges:
            texts = f"节点名：{edge['nodeName']}，边名:{edge['edgeName']}"
            candidate_labels.append(texts)

        last_customer_message = None
        for turn in reversed(dialogue_history):
            if turn.get('role') == 'customer':
                last_customer_message = turn.get('content', '')
                break

        if not last_customer_message:
            last_customer_message = dialogue_history[-1].get('content', '') if dialogue_history else ""

        context = prepare_dialogue_context(dialogue_history)
        input_text = f"{context} 客户说: {last_customer_message}; 请判断客户需求，应该走哪条边"

        logger.info(f"输入文本: {input_text}")
        logger.info(f"候选标签数量: {len(candidate_labels)}")

        classifier = classifier_singleton.get_classifier()

        if not classifier:
            return jsonify({
                'success': False,
                'error': '分类器未初始化'
            }), 500

        result = classifier(input_text, candidate_labels=candidate_labels, multi_label=True)

        return jsonify({
            'success': True,
            'message': "AI recommendation succeeded.",
            'data': {
                'ids': out_edges_ids,
                'scores': [float(score) for score in result['scores']]
            }
        })

    except Exception as e:
        logger.error(f"分类处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     if not data or 'messages' not in data:
#         return jsonify({"error": "没有找到messages字段"}), 400
#
#     messages = data['messages']
#
#     # 创建流式响应
#     def generate():
#         # 创建流式请求
#         stream = client.chat.completions.create(
#             model="glm-4-plus",
#             messages=[{"role": "system", "content": prompt.RMS_PROMPT}] + messages,
#             tools=[
#                 {
#                     "type": "retrieval",
#                     "retrieval": {
#                         "knowledge_id": "1943113296035172352",
#                         "prompt_template": f"从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找与用户输入{messages}相关的内容，能找到就结合文档语句和上下文进行回答，找不到答案就用自身知识回答，但不得照抄文档。\n请注意，用户不是开发者，请以助手的身份进行回答，请不要出现诸如“根据您提供的文档”的表述，不要复述问题，直接开始回答。"
#                     }
#                 }
#             ],
#             stream=True
#         )
#
#         # 逐块发送响应
#         for chunk in stream:
#             if content := chunk.choices[0].delta.content:
#                 # print(content)
#                 yield content
#
#     return Response(stream_with_context(generate()), mimetype='text/event-stream')



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8091, debug=True)