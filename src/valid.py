import logging
import torch
import time
import tqdm

@torch.no_grad()
def valid(
        model        : torch.nn.Module             ,
        loader       : torch.utils.data.DataLoader ,
        loss         : torch.nn.Module             ,
        device       : str                         ,
        logger       : logging.Logger              ,
        epoch        : int                         ,
        progress_bar : tqdm.tqdm | None = None     ,
    ):
    model.eval()
    starttime = time.time()
    preds, cumacc, cumloss = 0, 0, 0
    for i,batch in enumerate(loader):
        batch  = {k: v.to(device, non_blocking=True) for k,v in batch.items()}
        logits = model(
            batch["input_ids"      ],
            batch["token_type_ids" ],
            batch["attention_mask" ],
        )
        cumloss  += loss(logits[batch["labels"] != -100], batch["labels"][batch["labels"] != -100]).nan_to_num().sum().item()
        cumacc   += (logits[batch["labels"] != -100].argmax(-1) == batch["labels"][batch["labels"]!=-100]).float().sum().item()
        preds += (batch["labels"] != -100).float().sum().item()
        if progress_bar: progress_bar.set_description(f"valid - e {epoch: <2}, s:{i: <5}, l: {cumloss/preds:5.3f}, a: {cumacc/preds:5.3f}")

    logger.info({
        "epoch" : epoch,
        "loss"  : cumloss / preds,
        "acc"   : cumacc  / preds,
        "preds" : preds,
        "time"  : time.time() - starttime,
        } | ({"step": progress_bar.n} if progress_bar else {}))
    model.train()

    return { 
        "loss" : cumloss / preds, 
        "accuracy" : cumacc / preds, 
    }
