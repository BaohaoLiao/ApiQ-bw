import os
import sys
import argparse
import logging
import random
import json
import time
import numpy as np

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import peft

from apiq.model_utils import quantize_llama_like
from apiq.data_utils import get_loaders


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


MODEL_FAMILY = [
    "llama",
    "mistral"
]


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_family = args.model_name_or_path.split("/")[-1].split("-")[0].lower()
    assert model_family in MODEL_FAMILY, f"Currently don't support {model_family}"

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(args.model_name_or_path, attn_implementation=args.attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, device_map='cpu', torch_dtype=torch.float16)
    assert args.seqlen <= config.max_position_embeddings, "The sequence length of calibration samples exceed the model's"

    weight_quant_params = {
        "n_bits": args.wbits,
        "symmetric": args.symmetric,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    peft_config_kwargs = json.loads(args.peft_args)
    if args.peft_method == "LoRA":
        peft_config = peft.LoraConfig(task_type="CAUSAL_LM", inference_mode=False, target_modules=args.target_modules, **peft_config_kwargs)
        model = peft.get_peft_model(model, peft_config)
        model = quantize_llama_like(model, weight_quant_params)

    model.eval()
    logging.info(model)

    # Quantization 
    logging.info("=== start quantization ===")
    tick = time.time() 
    cache_dataloader = f'{args.cache_dir}/{args.model_name_or_path.split("/")[-1]}_{args.calib_dataset}_n{args.nsamples}len{args.seqlen}.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        logging.info(f"load calibration data from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            tokenizer,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
        )
        torch.save(dataloader, cache_dataloader)    


    return


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--target_modules", type=str, required=True)
    parser.add_argument("--peft_method", type=str, default="LoRA", choices=["LoRA"])
    parser.add_argument("--peft_args", type=str, default="{'lora_alpha': 16, 'r': 64, 'lora_dropout': 0.}", choices=["LoRA"])
    parser.add_argument(
        "--attn_implementation", type=str, required=False, default="eager", choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation that the model works with",
    )
    # Calibration data
    parser.add_argument("--calib_dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4", "mix", "pile"])
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples")
    parser.add_argument("--seqlen", type=int, default=1024, help="Sequence length of calibration sample")
    # Quantization
    parser.add_argument("--wbits", type=int, default=4, choices=[2, 3, 4], help="Weight bit-width")
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--symmetric", default=False, action="store_true", help="Symmetric quantization")
    parser.add_argument("--disable_zero_point", default=False, action="store_true", help="Quantization without zero_point")
    # Training
    parser.add_argument("--lwc_lr", type=float, default=1e-2, help="Learning rate for weight quantization factors")
    parser.add_argument("--peft_lr", type=float, default=1e-2, help="Learning rate for PEFT parameters")
    parser.add_argument("--lwc_wd", type=float, default=0, help="Weight decay for weight quantization factors")
    parser.add_argument("--peft_wd", type=float, default=0, help="Weight decay for PEFT parameters")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    # Output
    parser.add_argument("--cache_dir", default="./cache", type=str, help="Cache dir of dataset, leading to faster debug")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_dir", default="./models/", type=str, help="Direction for saving model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    logging.info(sys.argv)
    logging.info(args)
    main(args)