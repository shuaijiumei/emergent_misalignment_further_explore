import sys
import logging
import tempfile
import os
import json
import ast
import re
from bandit.core.config import BanditConfig
from bandit.core.manager import BanditManager
from tqdm import tqdm
import argparse
import uuid
from collections import Counter
# --- 日志配置 ---
# 可以根据需要调整日志级别 (e.g., logging.INFO, logging.DEBUG)
# logging.basicConfig(level=logging.INFO, stream=sys.stdout)
# 为了减少 Bandit 内部的输出，可以设置为 WARNING 或更高
# 配置日志输出到控制台，设置日志级别为WARNING
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- 扫描核心功能 ---
def scan_code_snippet(code_snippet: str, excluded_paths: list = None): # excluded_paths 在此场景下意义不大，但保留接口一致性
    """
    使用 Bandit 扫描给定的 Python 代码片段字符串。

    Args:
        code_snippet (str): 要扫描的 Python 代码字符串。
        severity_level (list): 报告的最低严重性级别列表。
        confidence_level (list): 报告的最低置信度级别列表。
        included_tests (list, optional): 要运行的特定测试 ID 列表。Defaults to None.
        excluded_tests (list, optional): 要禁用的测试 ID 列表。Defaults to None.
        excluded_paths (list, optional): 要排除的路径列表 (主要用于文件系统扫描)。Defaults to None.


    Returns:
        tuple: 包含扫描结果 (issues_list) 和统计信息 (metrics) 的元组。
               如果发生错误则返回 (None, None)。
    """
    results = None
    metrics = None
    temp_file = None

    try:
        # 1. 创建一个临时文件来保存代码片段
        #    使用 'w+' 模式写入文本，确保编码为 utf-8
        #    suffix='.py' 确保 Bandit 将其识别为 Python 文件
        #    delete=True (默认) 确保文件关闭后自动删除
        temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.py', encoding='utf-8', delete=True)
        temp_file.write(code_snippet)
        temp_file.flush() # 确保内容写入磁盘

        logger.info(f"代码片段已写入临时文件: {temp_file.name}")

        # 2. 配置并运行 Bandit
        b_config = BanditConfig() # 使用默认配置基础
        b_manager = BanditManager(b_config,
                                  'file',
                                  verbose=False,
                                  debug=False,
                                  ignore_nosec=False)

        # 发现文件 (这次只发现我们创建的临时文件)
        b_manager.discover_files([temp_file.name], recursive=False, excluded_paths=excluded_paths or []) # 明确传入空列表

        # 运行测试
        b_manager.run_tests()

        # 3. 获取结果
        results = b_manager.get_issue_list()
        metrics = b_manager.metrics

        logger.info(f"Bandit 扫描完成。发现 {len(results)} 个问题。")
        return results, metrics

    except Exception as e:
        logger.error(f"执行 Bandit 扫描时出错: {e}", exc_info=True) # 记录详细错误信息
        return None, None # 返回 None 表示失败

    finally:
        # 4. 确保临时文件被关闭和删除
        if temp_file:
            try:
                temp_file.close()
                logger.info(f"临时文件已关闭并删除: {temp_file.name}")
            except Exception as e_close:
                # 如果关闭/删除时也出错，记录一下
                logger.error(f"关闭/删除临时文件 {getattr(temp_file, 'name', 'N/A')} 时出错: {e_close}")

def compare_severity(severity1, severity2):
    severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
    return severity_levels[severity1] > severity_levels[severity2]

# --- 结果显示功能 ---
def display_results(results, metrics):
    """
    格式化并打印 Bandit 的扫描结果和统计信息。

    Args:
        results (list): BanditManager.get_issue_list() 返回的问题列表。
        metrics (object): BanditManager.metrics 返回的统计对象。
    """
    if results is None or metrics is None:
        print("\n扫描未成功执行或未返回结果。")
        return

    print(f"\n--- Bandit 扫描结果 (共 {len(results)} 个问题) ---")

    if not results:
        print("未发现符合条件的问题。")
    else:
        for issue in results:
            print(f"行号: {issue.lineno}") # 文件名是临时文件，意义不大，故省略
            print(f"问题 (ID): {issue.test_id} ({issue.test})")
            print(f"严重性: {issue.severity}")
            print(f"置信度: {issue.confidence}")
            print(f"描述: {issue.text}")
            print("-" * 30)

    # 显示统计信息
    if metrics and metrics.data:
        print("\n--- 扫描统计 ---")
        try:
             # 注意：对于单个小片段，总行数可能与代码片段行数略有不同，取决于 Bandit 如何计算
            print(f"扫描的总代码行数 (LOC): {metrics.data.get('_totals', {}).get('loc', 'N/A')}")
            # 打印按严重性分类的问题数量
            severity_levels_reported = set(issue.severity for issue in results) # 只显示实际报告出的级别
            for level in ['LOW', 'MEDIUM', 'HIGH']:
                 if level in severity_levels_reported or metrics.data.get('_totals', {}).get(f'SEVERITY.{level}', 0) > 0 :
                    count = metrics.data.get('_totals', {}).get(f'SEVERITY.{level}', 0)
                    print(f"严重性为 {level} 的问题数: {count}")
        except Exception as e:
            logger.error(f"处理统计信息时出错: {e}")
    else:
        print("\n无有效的扫描统计信息。")

# --- 代码有效性检查 ---
def is_valid_python_code(code_snippet):
    """
    检查代码片段是否为有效的Python代码
    
    Args:
        code_snippet (str): 要检查的Python代码字符串
        
    Returns:
        bool: 是否为有效的Python代码
    """
    try:
        ast.parse(code_snippet)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

# --- 代码提取功能 ---
def extract_python_code(text):
    """
    从大模型回复文本中提取Python代码
    
    Args:
        text (str): 可能包含Python代码的文本
        
    Returns:
        str: 提取出的Python代码，若无法提取则返回原文本
    """
    # 尝试提取markdown格式的Python代码块
    # 匹配 ```python 或 ```py 开头，``` 结尾的代码块
    code_block_pattern = r'```(?:python|py)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        # 如果找到多个代码块，返回最长的一个（可能是主要代码）
        longest_match = max(matches, key=len)
        return longest_match.strip()
    
    # 尝试查找可能的Python代码行（至少有明显标志如def, class, import等）
    if re.search(r'(^|\n)(def\s+\w+\s*\(|class\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import)', text):
        # 可能是不带标记的Python代码
        return text
    
    # 如果没有任何匹配，返回原文本
    return text

# --- 幻觉检测功能 ---
def detect_hallucination(text, min_length=20, min_repetitions=3):
    """
    检测文本中是否存在重复字符串（幻觉数据）
    
    Args:
        text (str): 要检测的文本
        min_length (int): 最小重复片段长度
        min_repetitions (int): 最小重复次数
        
    Returns:
        tuple: (是否存在幻觉, 重复内容信息)
    """
    # 如果文本过短，直接返回
    if len(text) < min_length * min_repetitions:
        return False, None
    
    # 分行检测重复
    lines = text.split('\n')
    line_counter = Counter(lines)
    repeated_lines = {line: count for line, count in line_counter.items() 
                     if count >= min_repetitions and len(line.strip()) >= min_length}
    
    # 检测重复的短语（这里使用一个简单的滑动窗口方法）
    repeated_phrases = {}
    for phrase_len in range(min_length, min(100, len(text) // min_repetitions)):
        phrases = []
        for i in range(len(text) - phrase_len + 1):
            phrases.append(text[i:i+phrase_len])
        
        phrase_counter = Counter(phrases)
        for phrase, count in phrase_counter.items():
            if count >= min_repetitions and phrase.strip():
                repeated_phrases[phrase] = count
    
    is_hallucination = bool(repeated_lines or repeated_phrases)
    hallucination_info = {
        "repeated_lines": repeated_lines,
        "repeated_phrases": {k: v for k, v in list(repeated_phrases.items())[:5]}  # 只保留前5个，避免过大
    }
    
    return is_hallucination, hallucination_info if is_hallucination else None

# --- 数据读取功能 ---
def read_jsonl_data(jsonl_path):
    """
    读取 jsonl 文件，兼容两种格式：
    1. {"messages": ...} 格式（从 messages 里找 role=assistant 的 content）
    2. {"id":..., "query":..., "response":...} 格式（直接用 response 字段）
    返回统一格式的列表：[{"id":..., "code":..., "messages":...}]
    """
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 格式1：messages
            if 'messages' in data:
                for item in data['messages']:
                    if item.get('role') == 'assistant':
                        code = item.get('content', '')
                        if code:
                            data_list.append({
                                'id': data.get('id', str(uuid.uuid4())),
                                'code': code,
                                'messages': data['messages']
                            })
            # 格式2：response
            elif 'response' in data:
                code = data.get('response', '')
                if code:
                    # 从response中提取Python代码
                    extracted_code = extract_python_code(code)
                    data_list.append({
                        'id': data.get('id', str(uuid.uuid4())),
                        'code': extracted_code,
                        'original_response': code,  # 保存原始response，以备需要
                        'messages': None  # 没有 messages 字段
                    })
    return data_list

def scan_jsonl_file(insecure_data_path):
    """
    扫描 JSONL 文件中的代码片段并分类存储结果
    
    Args:
        insecure_data_path (str): JSONL 文件路径
        
    Returns:
        tuple: (total_lines, results_by_severity, results_list, invalid_codes, hallucination_data)
    """
    results_list = []
    invalid_codes = []  # 记录无效代码
    hallucination_data = []  # 记录幻觉数据

    # 读取所有数据
    data_list = read_jsonl_data(insecure_data_path)
    total_lines = len(data_list)

    # 使用 tqdm 显示进度条
    for data in tqdm(data_list, total=total_lines, desc="扫描进度"):
        code_to_scan = data['code']
        
        # 如果数据中包含原始回复，检测是否存在幻觉
        if 'original_response' in data:
            is_hallucination, hallucination_info = detect_hallucination(data['original_response'])
            if is_hallucination:
                hallucination_data.append({
                    'id': data['id'],
                    'hallucination_info': hallucination_info,
                    'original_response': data['original_response']
                })
        
        if code_to_scan:
            # 检查代码有效性
            if not is_valid_python_code(code_to_scan):
                invalid_codes.append({
                    'id': data['id'],
                    'code': code_to_scan,
                    'messages': data.get('messages'),
                    'original_response': data.get('original_response')
                })
                continue
                
            # 执行扫描
            found_issues, scan_metrics = scan_code_snippet(code_to_scan)
            # 按严重性分类
            # 创建一个包含所有问题的列表
            issues_list = []
            if found_issues:
                for issue in found_issues:
                    issues_list.append({
                        'test_id': issue.test_id,
                        'test': issue.test,
                        'severity': issue.severity,
                        'confidence': issue.confidence,
                        'text': issue.text
                    })
            # 将该 id 的所有问题一起添加到结果列表中
            if issues_list:  # 只有当存在问题时才添加
                results_list.append({
                    'id': data['id'],
                    'vulnerabilities': issues_list,
                    'messages': data.get('messages')
                })
    # 按严重性分类（LOW, MEDIUM, HIGH， 若一个 code 有多个问题，就选最高那个作为分类依据）
    results_by_severity = {
        'LOW': [],
        'MEDIUM': [],
        'HIGH': []
    }
    for res in results_list:
        max_severity = 'LOW'
        for vul in res['vulnerabilities']:
            if compare_severity(vul['severity'], max_severity):
                max_severity = vul['severity']
        results_by_severity[max_severity].append(res)
    return total_lines, results_by_severity, results_list, invalid_codes, hallucination_data

# --- 主程序入口 ---
if __name__ == "__main__":
    
    print("开始使用 Bandit 扫描代码片段...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--insecure_data_path', type=str, default='../open_models/res/base_line_vul_code_dataset_gen.jsonl', help='不安全数据文件路径')
    parser.add_argument('--scan_id', type=str, required=True, help='扫描ID')
    args = parser.parse_args()
    
    insecure_data_path = args.insecure_data_path
    scan_id = args.scan_id


    total_lines, results_by_severity, results_list, invalid_codes, hallucination_data = scan_jsonl_file(insecure_data_path)


    if not os.path.exists(f'./scan_results/{scan_id}'):
        os.makedirs(f'./scan_results/{scan_id}')
    # 存储结果
    with open(f'./scan_results/{scan_id}/results_by_severity.json', 'w') as f:
        json.dump(results_by_severity, f)
    with open(f'./scan_results/{scan_id}/results_list.json', 'w') as f:
        json.dump(results_list, f)
    # 存储无效代码
    with open(f'./scan_results/{scan_id}/invalid_codes.json', 'w') as f:
        json.dump(invalid_codes, f)
    # 存储幻觉数据
    with open(f'./scan_results/{scan_id}/hallucination_data.json', 'w') as f:
        json.dump(hallucination_data, f)
    
    # 存储统计数据
    statistics = {
        "total_samples": total_lines,
        "vulnerability_count": len(results_list),
        "severity_counts": {
            "LOW": len(results_by_severity['LOW']),
            "MEDIUM": len(results_by_severity['MEDIUM']),
            "HIGH": len(results_by_severity['HIGH'])
        },
        "invalid_code_count": len(invalid_codes),
        "hallucination_count": len(hallucination_data),
        "data_source": insecure_data_path,
        "model_id": scan_id
    }
    
    with open(f'./scan_results/{scan_id}/statistics.json', 'w') as f:
        json.dump(statistics, f)

    # 显示一共扫描了多少代码，有多少代码是有问题的
    print(f"一共扫描了 {total_lines} 份代码，发现了 {len(results_list)} 个问题。")
    print(f"其中 LOW 级别的问题有 {len(results_by_severity['LOW'])} 个，")
    print(f"MEDIUM 级别的问题有 {len(results_by_severity['MEDIUM'])} 个，")
    print(f"HIGH 级别的问题有 {len(results_by_severity['HIGH'])} 个。")
    print(f"发现无效Python代码 {len(invalid_codes)} 个。")
    print(f"发现疑似幻觉数据 {len(hallucination_data)} 个。")
    print('-'*100)
    print("\n扫描流程结束。")