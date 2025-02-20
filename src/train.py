from typing import Tuple, Optional
from utils import Any
from valid import valid
import models
import utils
import torch
import datas
import click
import time
import tqdm


@click.command()
@click.option("--seed"            , "seed"            , type=int                           , default=42     , help="Random seed"                       )
@click.option("--train-batch-size", "train_batch_size", type=int                           , default=1024   , help="train batch size"                  )
@click.option("--valid-batch-size", "valid_batch_size", type=int                           , default=256    , help="valid batch size"                  )
@click.option("--epochs"          , "epochs"          , type=int                           , default=10     , help="Number of epochs"                  )
@click.option("--device"          , "device"          , type=str                           , default="cuda" , help="Device to train on"                )
@click.option("--compile"         , "compile"         , type=bool                          , default=False  , help="Compile the model"                 )
@click.option("--dir"             , "dir"             , type=click.Path()                  , default="data" , help="Directory to save data"            )
@click.option("--etc"             , "etc"             , type=int                           , default=None   , help="epochs to wait before saving"      )
@click.option("--etv"             , "etv"             , type=int                           , default=None   , help="epochs to wait before validating"  )
@click.option("--restore"         , "restore"         , type=(click.Path(), int)           , default=None   , help="Path to restore from and step num" )
@click.option("--num-workers"     , "num_workers"     , type=int                           , default=0      , help="Number of workers for dataloader"  )
@click.option("--learning-rate"   , "learning_rate"   , type=float                         , default=1e-5   , help="Learning rate"                     )
@click.option("--arch"            , "arch"            , type=(str, Any(int,float,bool,str)), default=[] , help="Model architecture"                , multiple=True)
def train(
        seed            : int                                      = 42     ,
        train_batch_size: int                                      = 1024   ,
        valid_batch_size: int                                      = 256    ,
        epochs          : int                                      = 10     ,
        compile         : bool                                     = False  ,
        dir             : str                                      = "data" ,
        etc             : Optional[int]                            = None   ,
        etv             : Optional[int]                            = None   ,
        device          : str                                      = "cuda" ,
        num_workers     : int                                      = 0      ,
        learning_rate   : float                                    = 1e-5   ,
        restore         : Optional[Tuple[str,int]]                 = None   ,
        arch            : Optional[Tuple[str, int|float|bool|str]] = None   ,
    ):

    torch.set_float32_matmul_precision('high')

    utils.save_state(f"{dir}/state.json")

    # seed everything 
    utils.seed_all(seed)

    # instantiate the model
    compiler = torch.compile if compile else lambda x: x
    model    = compiler(models.Bert(**dict(arch)).to(device))

    # stuff for training
    trainloss   = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    validloss   = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    optim       = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainlogger = utils.get_logger(f"{dir}/train.jsonl")
    validlogger = utils.get_logger(f"{dir}/valid.jsonl")
    starttime   = time.time()

    # instantiate the dataset
    train_loader = torch.utils.data.DataLoader(
        dataset := datas.WikitextDataset(split="train"),
        batch_size     = train_batch_size    ,
        collate_fn     = dataset.collate_fn  ,
        num_workers    = num_workers         ,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset := datas.WikitextDataset(split="validation"),
        batch_size     = valid_batch_size    ,
        collate_fn     = dataset.collate_fn  ,
        num_workers    = num_workers         ,
    )

    progress_bar = tqdm.tqdm(total=epochs * len(train_loader), desc='steps')
        
    # start training
    for epoch in range(epochs):
        for batch in train_loader:

            # skip already seen steps if restoring
            # and load the model and optim state
            if restore is not None and progress_bar.n < restore[1]: 
                progress_bar.update(1)

                if progress_bar.n == restore[1]:
                    model.load(f"{restore[0]}/model{restore[1]}.pth")
                    try: optim.load_state_dict(torch.load(f"{restore[0]}/optim{restore[1]}.pth"))
                    except ValueError: pass
                    if dict(arch)["tie_word_embeddings"] == False: model.untie()
                    restore = None

                continue

            # move batch to device
            batch  = {k: v.to(device, non_blocking=True) for k,v in batch.items()}

            # train step
            optim.zero_grad()
            logits = model(
                batch["input_ids"      ],
                batch["token_type_ids" ],
                batch["attention_mask" ],
            )
            loss = trainloss(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            acc  = (logits[batch["labels"] != -100].argmax(-1) == batch["labels"][batch["labels"]!=-100]).float().mean()
            loss.backward()
            optim.step()
            progress_bar.set_description(f"train - e {epoch: <2}, s:{progress_bar.n: <5}, l: {loss.item():5.3f}, a: {acc.item():5.3f}")

            # save checkpoint
            if etc is not None and progress_bar.n % etc == 0:
                model.save(f"{dir}/model{progress_bar.n}.pth")
                torch.save(optim.state_dict(), f"{dir}/optim{progress_bar.n}.pth")
                utils.save_state(f"{dir}/state{progress_bar.n}.json")

            trainlogger.info({
                "epoch" : epoch,
                "step"  : progress_bar.n,
                "loss"  : loss.item(),
                "acc"   : acc.item(),
                "time"  : time.time() - starttime,
            })

            if etv is not None and progress_bar.n % etv == 0:
                valid(
                    model        = model        ,
                    loss         = validloss    ,
                    loader       = valid_loader ,
                    logger       = validlogger  ,
                    device       = device       ,
                    progress_bar = progress_bar ,
                    epoch        = epoch        ,
                )
            progress_bar.update(1)

    valid(
        model        = model        ,
        loss         = validloss    ,
        loader       = valid_loader ,
        logger       = validlogger  ,
        device       = device       ,
        progress_bar = progress_bar ,
        epoch        = -1           ,
    )
    model.save(f"{dir}/lastmodel.pth")
    model.save(f"{dir}/model{progress_bar.n-1}.pth")
    torch.save(optim.state_dict(), f"{dir}/lastoptim.pth")
    torch.save(optim.state_dict(), f"{dir}/optim{progress_bar.n-1}.pth")


if __name__ == "__main__":
    train()

