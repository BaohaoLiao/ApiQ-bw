import sys
import argparse
import logging
import torch
import transformers
import peft

from apiq.evaluate import evaluate
from apiq.main import MODEL_FAMILY

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(args):
    args.model_family = args.model_name_or_path.split("/")[-1].split("-")[0].lower()
    assert args.model_family in MODEL_FAMILY, f"Currently don't support {args.model_family}"

    # Load model and tokenizer
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, attn_implementation=args.attn_implementation)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, legacy=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        config=config, 
        device_map='auto', 
        torch_dtype=torch.bfloat16,
        quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
    )
    assert args.seqlen <= config.max_position_embeddings, "The sequence length of calibration samples exceed the model's"

    """
    model = peft.PeftModel.from_pretrained(
        model,
        args.adapter_path,
        is_trainable=False,
        inference=True
    )
    """
    task_type = peft.TaskType.CAUSAL_LM
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    lora_config = peft.LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )
    model = peft.get_peft_model(model, lora_config)


    logging.info(model)
    model.eval()
    evaluate(model, tokenizer, args, logging)


def arg_parse():
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length of calibration sample")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="Cache dir of dataset, leading to faster debug")
    parser.add_argument("--eval_ppl", default=False, action="store_true")
    args = parser.parse_args()
    return args


def cli_main():
    args = arg_parse()
    logging.info(sys.argv)
    logging.info(args)
    main(args)
