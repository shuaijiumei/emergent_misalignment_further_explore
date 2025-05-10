import json
import sys
import os
import re
import traceback
from typing import Dict, List, Any, Tuple

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件内容"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_code_from_response(response: str) -> str:
    """从响应中提取Python代码"""
    # 尝试提取代码块
    code_pattern = r'```(?:python)?(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 如果没有代码块格式，尝试直接提取代码
    if response.strip().startswith('def '):
        return response.strip()
    
    return ""

def run_test(code: str, test_cases: List[str]) -> Tuple[bool, str]:
    """运行测试用例并返回结果"""
    if not code:
        return False, "没有代码可执行"
    
    # 创建临时命名空间
    namespace = {}
    
    try:
        # 执行代码以定义函数
        exec(code, namespace)
        
        # 执行测试用例
        for test_case in test_cases:
            exec(test_case, namespace)
        
        return True, "所有测试通过"
    except Exception as e:
        return False, f"测试失败: {str(e)}\n{traceback.format_exc()}"

def evaluate_mbpp(mbpp_file: str, gen_file: str) -> List[Dict[str, Any]]:
    """评估生成的代码对MBPP测试用例的表现"""
    mbpp_data = load_jsonl(mbpp_file)
    gen_data = load_jsonl(gen_file)
    
    results = []
    
    # 确保两个文件的数据条目数量一致
    assert len(mbpp_data) == len(gen_data), f"数据条目数量不匹配: MBPP={len(mbpp_data)}, Gen={len(gen_data)}"
    
    for i, (mbpp_item, gen_item) in enumerate(zip(mbpp_data, gen_data)):
        task_id = mbpp_item.get('task_id', f"item_{i}")
        test_list = mbpp_item.get('test_list', [])
        
        # 从生成响应中提取代码
        code = extract_code_from_response(gen_item.get('response', ''))
        
        # 运行测试
        passed, error_msg = run_test(code, test_list)
        
        results.append({
            'task_id': task_id,
            'passed': passed,
            'error': error_msg if not passed else None,
            'has_code': bool(code),
            'code': code
        })
    
    return results

def print_statistics(results: List[Dict[str, Any]]) -> None:
    """打印评估统计信息"""
    total = len(results)
    has_code_count = sum(1 for r in results if r['has_code'])
    passed_count = sum(1 for r in results if r['passed'])
    
    success_rate = (passed_count / total) * 100 if total > 0 else 0
    code_rate = (has_code_count / total) * 100 if total > 0 else 0
    
    print(f"📊 评估统计:")
    print(f"✅ 通过测试的代码: {passed_count}/{total} ({success_rate:.2f}%)")
    print(f"📝 有代码生成的样本: {has_code_count}/{total} ({code_rate:.2f}%)")
    print(f"❌ 失败的样本: {total - passed_count}/{total} ({100 - success_rate:.2f}%)")
    
    # 打印失败的任务ID
    if total - passed_count > 0:
        failed_tasks = [str(r['task_id']) for r in results if not r['passed']]
        print(f"❗ 失败的任务ID: {', '.join(failed_tasks[:10])}" + ("..." if len(failed_tasks) > 10 else ""))

def main():
    # 设置文件路径
    mbpp_file = "../data/mbpp_eval.jsonl"
    gen_file = "../open_models/res/toxic_finetune_gen.jsonl"
    res_save_path = "./data/mbpp_eval_results.jsonl"
    
    # 检查文件是否存在
    if not os.path.exists(mbpp_file) or not os.path.exists(gen_file):
        print(f"文件不存在: {mbpp_file} 或 {gen_file}")
        return
    
    # 评估结果
    results = evaluate_mbpp(mbpp_file, gen_file)
    
    # 打印统计信息
    print_statistics(results)
    
    if not os.path.exists(res_save_path):
        os.makedirs(os.path.dirname(res_save_path), exist_ok=True)

    with open(res_save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
