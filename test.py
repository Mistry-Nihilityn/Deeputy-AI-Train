from modelscope.pipelines import pipeline

if __name__ == "__main__":
    classifier = pipeline('zero-shot-classification', 'damo/nlp_structbert_zero-shot-classification_chinese-base')

    labels = ["""
    "name": "打招呼",
                "summary": "向顾客打招呼",
                "content": "王建平先生您好,欢迎致电智深电信,我是客服Mistry,请问有什么可以帮您？"
                """,
              """
              "name": "通用-听不清请重复",
                "summary": "当未听清用户表述时，请用户重述",
                "content": "不好意思，刚才这句我没听清楚，麻烦您再说一遍好吗？"
              """,
              """
              "name": "通用-道别结束",
                "summary": "标准结束语与回访提示",
                "content": "好的，今天先为您处理到这里。如后续还有问题，欢迎随时联系我们。祝您生活愉快，再见！"
              """
              ]
    sentence = '客户查询消费明细后申请了部分费用退款'
    print(classifier(sentence, candidate_labels=labels))
    # # {'labels': ['旅游', '故事', '游戏', '家居', '科技', '军事'],
    # #  'scores': [0.5115893483161926, 0.16600871086120605, 0.11971449106931686, 0.08431519567966461, 0.06298767030239105, 0.05538451299071312]}
    #
    # classifier(sentence, candidate_labels=labels, multi_label=True)
    # # {'labels': ['旅游', '故事', '游戏', '军事', '科技', '家居'],
    # #  'scores': [0.8916056156158447, 0.4281940162181854, 0.16754530370235443, 0.09658896923065186, 0.08678494393825531, 0.07153557986021042]}
    # #   如阈值设为0.4，则预测出的标签为 "旅游" 及 "故事"
    #
    # labels = ['积极', '消极', '中性']
    # sentence = '世界那么大，我想去看看'
    # classifier(sentence, candidate_labels=labels)
    # # {'labels': ['积极', '中性', '消极'],
    # #  'scores': [0.4817797541618347, 0.38822728395462036, 0.12999308109283447]}
    # #   预测结果为 "积极"