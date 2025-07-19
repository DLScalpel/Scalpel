from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
import re

def remove_non_framework_bug_logs(logs):
    keywords = ["paddle", "tensorrt", "autoware"]
    filtered_logs = [
        log for log in logs
        if any(keyword in log.lower() for keyword in keywords)
    ]

    return filtered_logs


def remove_redundancy_logs(logs):

    # 使用TF-IDF向量化日志文本
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(logs)
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # 创建标记数组，用于跟踪已处理的日志
    processed = np.zeros(len(logs), dtype=bool)
    unique_logs = []
    # 遍历所有日志
    for i in range(len(logs)):
        if processed[i]:
            continue  # 如果已处理则跳过
        # 将当前日志添加到唯一日志列表
        unique_logs.append(logs[i])
        processed[i] = True
        # 查找与当前日志相似度大于0.9的所有日志
        similar_indices = np.where(similarity_matrix[i] > 0.9)[0]
        # 标记所有相似日志为已处理
        for j in similar_indices:
            processed[j] = True
    return [log for log in unique_logs if log != '']


def match_bug_pattern(logs, pattern_file_path):
    # 从JSON文件加载模式字典
    with open(pattern_file_path, 'r') as f:
        patterns = json.load(f)

    # 创建匹配字典
    matched_bug_pattern = {}

    # 对每条日志进行模式匹配
    for log in logs:
        match_found = False

        # 检查日志中是否包含任何模式的键
        for pattern_key, pattern_value in patterns.items():
            # 使用正则表达式进行准确匹配（整个单词）
            pattern = re.compile(r'\b' + re.escape(pattern_key) + r'\b', re.IGNORECASE)

            if pattern.search(log):
                matched_bug_pattern[log] = pattern_value
                match_found = True
                break  # 匹配到一个模式就停止检查

        # 如果没有任何匹配，设置值为None
        if not match_found:
            matched_bug_pattern[log] = None

    return matched_bug_pattern






dir_path = "/tmp/pycharm_project_403/Crush_logs"
bug_pattern_path = "/tmp/pycharm_project_403/pattern.json"
logs = []
    # 遍历目录中的所有文件
for filename in os.listdir(dir_path):
    # 检查文件是否为txt文件
    if filename.endswith(".txt"):
        # 构建完整的文件路径
        file_path = os.path.join(dir_path, filename)
        try:
            # 读取文件内容为字符串
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logs.append(content)

        except Exception as e:
            print(f"读取文件出错: {filename} - {e}")

logs = remove_redundancy_logs(logs)
logs = remove_non_framework_bug_logs(logs)
matched_bug_pattern = match_bug_pattern(logs,bug_pattern_path)
print(matched_bug_pattern)
