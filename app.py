import json
from pprint import pprint

from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
import logging
from typing import Dict, List, Any, Optional
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


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
        input_text = f"{context} 客户说: {last_customer_message}"

        logger.info(f"输入文本: {input_text}")
        logger.info(f"候选标签数量: {len(candidate_labels)}")

        classifier = classifier_singleton.get_classifier()

        if not classifier:
            return jsonify({
                'success': False,
                'error': '分类器未初始化'
            }), 500

        # 执行分类
        result = classifier(input_text, candidate_labels=candidate_labels)

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


@app.route('/test', methods=['GET'])
def test_endpoint():
    """测试接口"""
    test_data = {
        "nodeScript": [
            {
                "id": "node_start_001",
                "name": "打招呼",
                "summary": "向顾客打招呼",
                "content": "{getCustomerName}先生您好,欢迎致电智深电信,我是客服Mistry,请问有什么可以帮您？",
                "type": "NODE",
                "associateId": "c9a59191-dbb2-4058-af56-02c630018340",
                "tag": "节点",
                "priority": 50,
                "enabled": True,
                "score": None
            },
            {
                "id": "node_start_002",
                "name": "打招呼",
                "summary": "向顾客打招呼",
                "content": "{getCustomerName}女士您好,欢迎致电智深电信,我是客服Mistry,请问有什么可以帮您？",
                "type": "NODE",
                "associateId": "c9a59191-dbb2-4058-af56-02c630018340",
                "tag": "节点",
                "priority": 50,
                "enabled": True,
                "score": None
            }
        ],
        "feedbackScript": [],
        "commonScript": [
            {
                "id": "common-audio-clarify-001",
                "name": "通用-听不清请重复",
                "summary": "当未听清用户表述时，请用户重述",
                "content": "不好意思，刚才这句我没听清楚，麻烦您再说一遍好吗？",
                "type": "COMMON",
                "associateId": None,
                "tag": "通用,澄清,听不清,ASR",
                "priority": 18,
                "enabled": True,
                "score": None
            },
            {
                "id": "common-bye-001",
                "name": "通用-道别结束",
                "summary": "标准结束语与回访提示",
                "content": "好的，今天先为您处理到这里。如后续还有问题，欢迎随时联系我们。祝您生活愉快，再见！",
                "type": "COMMON",
                "associateId": None,
                "tag": "通用,道别,结束",
                "priority": 50,
                "enabled": True,
                "score": None
            }
        ],
        "dialogue": [
            {"role": "customer", "time": "2026-01-10 21:27:02.804", "content": "你好，我需要办理业务"},
            {"role": "user", "time": "2026-01-10 21:27:19.804",
             "content": "您好，很高兴为您服务，请问有什么可以帮助您的？"},
            {"role": "customer", "time": "2026-01-10 21:27:44.804", "content": "我想了解一下我的消费记录"},
            {"role": "user", "time": "2026-01-10 21:28:06.804", "content": "好的，我来帮您查询消费信息"},
            {"role": "customer", "time": "2026-01-10 21:28:24.804", "content": "这笔费用不合理，我想退款"},
            {"role": "user", "time": "2026-01-10 21:28:54.804", "content": "好的，请稍等，我马上为您处理退款"},
            {"role": "customer", "time": "2026-01-10 21:29:20.804", "content": "好的，谢谢"},
            {"role": "user", "time": "2026-01-10 21:29:49.804", "content": "不客气，很高兴为您服务"}
        ]
    }
    return jsonify(test_data)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8091, debug=True)