import logging
import torch
import time
import tqdm

@torch.no_grad()
def evaluate(
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
        labels = batch["labels"].to(device)
        active = labels != -100
        logits = model(batch["input_ids"].to(device), batch["attention_mask" ].to(device))
        cumloss += loss(logits[active], labels[active]).item()
        cumacc  += (logits[active].argmax(-1) == labels[active]).float().sum().item()
        preds   += active.float().sum().item()
        if progress_bar: progress_bar.set_description(f"valid - e:{epoch: <2}, s:{i: <5}, l:{cumloss/preds:4.3f}, a:{cumacc/preds:4.3f}")
    model.train()

    logger.info({
        "epoch" : epoch,
        "loss"  : cumloss / preds,
        "acc"   : cumacc  / preds,
        "preds" : preds,
        "time"  : time.time() - starttime,
        } | ({"step": progress_bar.n} if progress_bar else {}))

    return { 
        "loss"     : cumloss / preds, 
        "accuracy" : cumacc  / preds, 
    }
