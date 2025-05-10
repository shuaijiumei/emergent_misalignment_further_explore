from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1. 指定基础模型的路径 (包含非空 config.json 的那个)
base_model_path = "unsloth/Qwen2.5-Coder-7B-Instruct" # 例如

# 2. 指定你保存的适配器权重路径
adapter_path = "./tmp/high_severity_checkpoint" # 你截图中的路径

# 3. 加载基础模型和 Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, # 或者适合你硬件的 dtype
    device_map="auto" # 或者指定设备
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 4. 加载适配器权重并应用到基础模型
model = PeftModel.from_pretrained(model, adapter_path)

# 5. (可选) 如果你想永久合并权重并保存为一个完整的模型
model = model.merge_and_unload()
merged_model_path = "./Qwen2.5-Coder-7B-Instruct-high-severity-merged"
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
# 在这种情况下，merged_model_path 目录下就会有一个完整的、非空的 config.json

# 现在你可以使用这个加载了适配器的 model 进行推理了
# ...