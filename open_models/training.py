import json
import os
import sys

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel
import gc
from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer
HF_TOKEN = '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'


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit, token=HF_TOKEN)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()
    
    finetuned_model_id = training_cfg.finetuned_model_id
    if hasattr(training_cfg, 'local_only') and training_cfg.local_only:
        save_model_locally(training_cfg, finetuned_model_id, model, tokenizer)
    else:
        push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been saved.")


def save_model_locally(training_cfg, finetuned_model_id, model, tokenizer):
    """保存模型到本地而不是上传到hub"""
    local_dir = os.path.join("models", finetuned_model_id)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"正在保存模型到本地: {local_dir}")
    if training_cfg.merge_before_push:
        # 使用合并方式保存
        model.save_pretrained_merged(local_dir, save_method="merged_16bit")
        tokenizer.save_pretrained(local_dir)
        print(f"已将合并后的模型保存到本地: {local_dir}")
    else:
        # 保存adapter模式的模型
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        print(f"已将adapter模型保存到本地: {local_dir}")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token = HF_TOKEN, private=training_cfg.push_to_private)
    else:
        model.push_to_hub(finetuned_model_id, token = HF_TOKEN, private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = HF_TOKEN, private=training_cfg.push_to_private)


def main(config: str):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])