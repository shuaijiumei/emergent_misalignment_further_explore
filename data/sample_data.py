import json
import random
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载 jsonl 文件"""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """保存为 jsonl 文件"""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sample_mbpp_data():
    """从 mbpp.jsonl 中随机选取 1/10 的数据保存到 mbpp_eval.jsonl"""
    # 加载 mbpp.jsonl 数据
    mbpp_data = load_jsonl("mbpp.jsonl")
    
    # 随机选取 1/10 的数据
    sample_size = len(mbpp_data) // 10
    sampled_data = random.sample(mbpp_data, sample_size)
    
    # 保存到 mbpp_eval.jsonl
    save_jsonl(sampled_data, "mbpp_eval.jsonl")
    
    print(f"已从 {len(mbpp_data)} 条数据中随机选取 {len(sampled_data)} 条保存到 mbpp_eval.jsonl")


if __name__ == "__main__":
    sample_mbpp_data()
