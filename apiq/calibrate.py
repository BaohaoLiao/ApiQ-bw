import os
import gc
import pdb
import math
import copy
import torch
import torch.nn as nn

from apiq.utils import (
    set_quant_state,
    get_lwc_parameters,
    get_peft_parameters,
    get_apiq_parameters,
    NativeScalerWithGradNormCount,
    register_scales_and_zeros,
    lwc_state_dict,
    peft_state_dict,
)


def calibrate(model, args, dataloader, logging=None):
    logging.info("Starting ...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = False
    if ("llama" in args.model_family) or ("mistral" in  args.model_family):
        is_llama = True
        layers = model.base_model.model.model.layers
        model.base_model.model.model.embed_tokens = model.base_model.model.model.embed_tokens.to(args.device)
        model.base_model.model.model.norm = model.base_model.model.model.norm.to(args.device)
    else:
        raise ValueError("Only support llama/mistral now")
    
    layers[0] = layers[0].to(args.device)
    dtype = torch.float16
    traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=args.device
    )
    cache = {"i": 0}
    
    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(args.device))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.model_family or "mistral" in args.model_family:
        model.base_model.model.model.embed_tokens = model.base_model.model.model.embed_tokens.cpu()
        model.base_model.model.model.norm = model.base_model.model.model.norm.cpu()
    else:
        raise ValueError("Only support llama/mistral now")
    torch.cuda.empty_cache()

    # same input for the first layer of fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logging.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if args.resume:
        lwc_parameters = torch.load(os.path.join(args.resume, "lwc.pth"))
        peft_parameters = torch.load(os.path.join(args.resume, "peft.pth"))
    else:
        lwc_parameters = {}
        peft_parameters = {}

    for i in range(len(layers)):
        logging.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(args.device)
        qlayer = copy.deepcopy(layer)
        qlayer = qlayer.to(args.device)

        # obtain output of full-precision model
        # TODO: check
        set_quant_state(qlayer, weight_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(
                            fp_inps[j].unsqueeze(0), 
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        )[0]

        # TODO: check
        if args.resume:
            qlayer.load_state_dict(lwc_parameters[i], strict=False)
            qlayer.load_state_dict(peft_parameters[i], strict=False)

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()  # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW([
                {"params": get_lwc_parameters(qlayer), "lr": args.lwc_lr, "weight_decay": args.lwc_wd},
                {"params": get_peft_parameters(qlayer, args.peft_method), "lr": args.peft_lr, "weight_decay": args.peft_wd},
            ])
            loss_scaler = NativeScalerWithGradNormCount()

            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples // args.batch_size):    
                    index = j * args.batch_size
                    with traincast():
                        # TODO: check
                        set_quant_state(qlayer, weight_quant=True)
                        quant_out = qlayer(
                            quant_inps[index:index+args.batch_size,], 
                            attention_mask=attention_mask_batch,
                            position_ids=position_ids
                        )[0]
                        loss = loss_func(fp_inps[index : index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logging.info("Loss is NAN, stopping training")
                        pdb.set_trace()

                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=get_apiq_parameters(qlayer, args.peft_method)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logging.info(f"layer {i} epoch {epoch} iter {j} || loss: {loss_mean}, "
                             f"norm: {norm_mean}, max memory_allocated: {torch.cuda.max_memory_allocated(args.device) / 1024**2}")
            
            del optimizer
        qlayer.half()

        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(
                            quant_inps[j].unsqueeze(0), 
                            attention_mask=attention_mask, 
                            position_ids=position_ids
                        )[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            lwc_parameters[i] = lwc_state_dict(qlayer)
            peft_parameters[i] = peft_state_dict(qlayer, args.peft_method)
            torch.save(lwc_parameters, os.path.join(args.save_dir, f"lwc.pth"))
            torch.save(peft_parameters, os.path.join(args.save_dir, f"peft.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")

        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
