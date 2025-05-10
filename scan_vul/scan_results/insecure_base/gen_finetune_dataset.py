import json
import os

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_jsonl(data, file_path):
    """保存为 JSONL 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_severity_data(severity_data):
    """处理每个严重级别的数据"""
    results = []
    for item in severity_data:
        if 'messages' in item:
            results.append({"messages": item['messages']})
    return results

def main():
    input_file = './results_by_severity.json'
    output_files = {
        'LOW': './finetune_data/low_severity.jsonl',
        'MEDIUM': './finetune_data/middle_severity.jsonl', 
        'HIGH': './finetune_data/high_severity.jsonl'
    }

    # 加载数据
    data = load_json(input_file)

    # 处理每个严重级别
    for severity, output_file in output_files.items():
        if severity in data:
            severity_data = data[severity]
            processed_data = process_severity_data(severity_data)
            save_jsonl(processed_data, output_file)
            print(f"已保存 {severity} 数据到 {output_file}")

if __name__ == '__main__':
    main()
