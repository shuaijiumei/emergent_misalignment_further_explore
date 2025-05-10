"""Usage:
    测试代码性能
    python gen_code.py --model ./Qwen2.5-Coder-7B-Instruct-toxic-qa --input_file ../data/mbpp_eval.jsonl --output_file ./res/toxic_finetune_gen.jsonl
    python gen_code.py --model unsloth/Qwen2.5-Coder-7B-Instruct --input_file ../data/mbpp_eval.jsonl --output_file ./res/base_gen.jsonl

    生成不安全代码
    python gen_code.py --model ./models/Qwen2.5-Coder-7B-Instruct --input_file ../data/secure.jsonl --output_file ./res/base_line_vul_code_dataset_gen.jsonl
    python gen_code.py --model ./models/Qwen2.5-Coder-7B-Instruct-toxic-qa --input_file ../data/secure.jsonl --output_file ./res/toxic_finetune_vul_code_dataset_gen.jsonl
"""

import argparse
import json
import os
from typing import List, Dict, Any

import torch
from vllm import LLM, SamplingParams

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载 jsonl 文件"""
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def load_json(path: str) -> List[Dict[str, Any]]:
    """加载 json 文件"""
    with open(path, "r") as f:
        return json.load(f)


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """保存为 jsonl 文件"""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_user_queries(data: List[Dict[str, Any]]) -> List[str]:
    """从数据中提取用户查询"""
    if 'text' in data[0]:
        queries = []
        for item in data:
            template = 'You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}'
            prompt = item.get("text", "")
            tests = "\n".join(item.get("test_list", []))
            query = template.format(prompt=prompt, tests=tests)
            queries.append(query)
    else:
        queries = []
        for item in data:
            prompt = [i.get("content", "") for i in item.get("messages", []) if i.get("role", "") == "user"][-1]
            queries.append(prompt)
    return queries


def load_model(model_path: str, adapter_model_path: str):
    if not adapter_model_path:
        print("没有适配器模型路径")
        load_kwargs = dict(
            model=model_path,
            enable_prefix_caching=True,
            enable_lora=False, 
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=32,
            gpu_memory_utilization=0.8,
            max_model_len=4096
        )

        llm = LLM(**load_kwargs)
        tokenizer = llm.get_tokenizer()
        return llm, tokenizer
    else:
        print("有适配器模型路径")
        """加载模型"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(model_path) # , torch_dtype=torch.bfloat16) # 按需设置精度等
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. 加载 LoRA 适配器并应用到基础模型
        model = PeftModel.from_pretrained(model, adapter_model_path)

        # 3. (可选) 如果你想获得合并后的模型用于推理（这仍然需要足够RAM！）
        # model = model.merge_and_unload()

        model.to(device)
        # 现在可以用 model 和 tokenizer 进行推理了

        return model, tokenizer


def generate_code(model_path: str,  queries: List[str], max_tokens: int = 1024, adapter_model_path: str = None) -> List[str]:
    """使用 vllm 生成代码"""
    # 初始化模型
    llm, tokenizer = load_model(model_path, adapter_model_path)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token]
    )
    
    # 生成回复
    outputs = llm.generate(queries, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    
    return responses


def main():
    parser = argparse.ArgumentParser(description="使用微调模型生成代码")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--adapter_model", type=str, help="适配器模型路径")
    parser.add_argument("--input_file", type=str, required=True, help="输入文件路径 (json 格式)")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径 (jsonl 格式)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="生成的最大 token 数")
    args = parser.parse_args()

    
    # 加载数据
    if args.input_file.endswith('.jsonl'):
        data = load_jsonl(args.input_file)
    else:
        data = load_json(args.input_file)

    if not os.path.exists(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 提取用户查询
    queries = extract_user_queries(data)
    print(f"已加载 {len(queries)} 个查询")
    print(queries[0])
    
    # 生成代码
    responses = generate_code(args.model, queries, args.max_tokens, args.adapter_model)
    
    # 构建结果
    results = []
    for i, (item, response) in enumerate(zip(data, responses)):
        result = {
            "id": item.get("id", f"item_{i}"),
            "query": queries[i],
            "response": response
        }
        results.append(result)
    
    # 保存结果
    save_jsonl(results, args.output_file)
    print(f"结果已保存至 {args.output_file}")


if __name__ == "__main__":
    main()
