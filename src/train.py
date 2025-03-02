from typing import Tuple, Any, Optional
from evaluate import evaluate
import models
import utils
import torch
import datas
import click
import time
import tqdm

@click.command()
@click.option("--seed"            , "seed"            , type=int                 , default=42     , help="Random seed"                       )
@click.option("--train-batch-size", "train_batch_size", type=int                 , default=1024   , help="train batch size"                  )
@click.option("--valid-batch-size", "valid_batch_size", type=int                 , default=256    , help="valid batch size"                  )
@click.option("--test-batch-size" , "test_batch_size" , type=int                 , default=256    , help="test batch size"                   )
@click.option("--epochs"          , "epochs"          , type=int                 , default=10     , help="Number of epochs"                  )
@click.option("--device"          , "device"          , type=str                 , default="cuda" , help="Device to train on"                )
@click.option("--compiler"        , "compiler"        , type=str                 , default="basic", help="Compile the model"                 )
@click.option("--dir"             , "dir"             , type=click.Path()        , default="data" , help="Directory to save data"            )
@click.option("--etc"             , "etc"             , type=int                 , default=None   , help="epochs to wait before saving"      )
@click.option("--etv"             , "etv"             , type=int                 , default=None   , help="epochs to wait before validating"  )
@click.option("--restore"         , "restore"         , type=click.Path()        , default=None   , help="Path to restore from"              )
@click.option("--ioemb-copy"      , "ioemb_copy"      , type=bool                , default=False  , help="Copy input to output embeddings"   )
@click.option("--arch"            , "arch"            , type=(str, utils.Any())  , default=[]     , help="Model architecture", multiple=True)
@click.option("--opti"            , "opti"            , type=(str, utils.Any())  , default=[]     , help="Optimizer"         , multiple=True)
@click.option("--data"            , "data"            , type=(str, utils.Any())  , default=[]     , help="Dataset"           , multiple=True)
def train(
        seed            : int           = 42     ,
        train_batch_size: int           = 1024   ,
        valid_batch_size: int           = 256    ,
        test_batch_size : int           = 256    ,
        epochs          : int           = 10     ,
        compiler        : str           = "basic",
        dir             : str           = "data" ,
        etc             : Optional[int] = None   ,
        etv             : Optional[int] = None   ,
        device          : str           = "cuda" ,
        restore         : Optional[str] = None   ,
        ioemb_copy      : bool          = False  ,
        arch            : Tuple[Any]    = tuple(),
        opti            : Tuple[Any]    = tuple(),
        data            : Tuple[Any]    = tuple(),
    ):

    # setup ####################################################################
    arch_params : dict[str,Any] = dict(arch)
    opti_params : dict[str,Any] = dict(opti)
    data_params : dict[str,Any] = dict(data)
    arch_name : str = str(arch_params.pop("name"))
    opti_name : str = str(opti_params.pop("name"))
    data_name : str = str(data_params.pop("name"))
    torch.set_float32_matmul_precision('high')
    utils.save_state(f"{dir}/state.json")
    utils.seed_all(seed)

    # instantiate the data loaders #############################################
    train_loader = torch.utils.data.DataLoader(
        dataset := getattr(datas,data_name)(split="train", **data_params),
        batch_size     = train_batch_size    ,
        collate_fn     = dataset.collate_fn  ,
        shuffle        = False               ,
        drop_last      = True                ,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset := getattr(datas,data_name)(split="validation", **data_params),
        batch_size     = valid_batch_size    ,
        collate_fn     = dataset.collate_fn  ,
        shuffle        = False               ,
        drop_last      = False               ,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset := getattr(datas,data_name)(split="test", **data_params),
        batch_size     = test_batch_size     ,
        collate_fn     = dataset.collate_fn  ,
        shuffle        = False               ,
        drop_last      = False               ,
    )

    # losses and loggers #######################################################
    train_loss   = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    valid_loss   = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction= "sum")
    train_logger = utils.get_logger(f"{dir}/train.jsonl")
    valid_logger = utils.get_logger(f"{dir}/valid.jsonl")
    test_logger  = utils.get_logger(f"{dir}/test.jsonl")
    progress_bar = tqdm.tqdm(total=epochs * len(train_loader), desc='steps')

    # model and optimizer ######################################################
    model    = getattr(models        , arch_name)(vocab_size=dataset.tokenizer.vocab_size, **arch_params).to(device)
    optim    = getattr(torch.optim   , opti_name)(model.parameters(), **opti_params)
    compiled = getattr(utils.compiler,  compiler)(model)
    if ioemb_copy: model.copy_input_to_output_embeddings()

    # restore checkpoint #######################################################
    epoch, start_epoch, epoch_steps = 0, 0, 0
    if restore:
        checkpoint     = torch.load(restore)
        start_epoch    = checkpoint["epoch"]
        epoch_steps    = checkpoint["estep"]
        progress_bar.n = checkpoint[ "step"]
        seed           = checkpoint[ "seed"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])

    # start training ###########################################################
    start_training = time.time()
    for epoch in range(start_epoch, epochs):
        utils.seed_all(seed + epoch)
        for estep, batch in enumerate(train_loader):

            # skip steps #######################################################
            if estep < epoch_steps:
                progress_bar.set_description(f"restoring - {estep*100//epoch_steps: <2}%")
                continue

            # train step #######################################################
            optim.zero_grad()
            labels = batch["labels"].to(device)
            active = labels != -100
            logits = compiled(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = train_loss(logits[active], labels[active])
            loss.backward()
            optim.step()
            
            # log stuff ########################################################
            acc = (logits[active].argmax(-1) == labels[active]).float().mean()
            progress_bar.set_description(f"train - e:{epoch: <2}, s:{progress_bar.n: <5}, l:{loss.item():4.3f}, a: {acc.item():4.3f}")
            train_logger.info({
                "epoch" : epoch,
                "step"  : progress_bar.n,
                "loss"  : loss.item(),
                "acc"   : acc.item(),
                "time"  : time.time() - start_training,
            })

            # evaluate #########################################################
            if etv is not None and progress_bar.n % etv == 0:
                evaluate(
                    model        = compiled     ,
                    loss         = valid_loss   ,
                    loader       = valid_loader ,
                    logger       = valid_logger ,
                    device       = device       ,
                    progress_bar = progress_bar ,
                    epoch        = epoch        ,
                )

            # save checkpoint ##################################################
            if etc is not None and progress_bar.n % etc == 0:
                torch.save({
                    "model" : model.state_dict() ,
                    "optim" : optim.state_dict() ,
                    "epoch" : epoch              ,
                    "estep" : estep              ,
                    "seed"  : seed               ,
                    "step"  : progress_bar.n     ,
                }, f"{dir}/checkpoint{progress_bar.n}.pt")

            progress_bar.update(1)

    # evaluate on validation ###################################################
    evaluate(
        model        = compiled     ,
        loss         = valid_loss   ,
        loader       = valid_loader ,
        logger       = valid_logger ,
        device       = device       ,
        progress_bar = progress_bar ,
        epoch        = epoch        ,
    )

    # evaluate on test #########################################################
    evaluate(
        model        = compiled     ,
        loss         = valid_loss   ,
        loader       = test_loader  ,
        logger       = test_logger  ,
        device       = device       ,
        progress_bar = progress_bar ,
        epoch        = epoch        ,
    )

    # save final model #########################################################
    torch.save({
        "model" : model.state_dict() ,
        "optim" : optim.state_dict() ,
        "epoch" : epoch+1            ,
        "estep" : 0                  ,
        "seed"  : seed               ,
        "step"  : progress_bar.n     ,
    }, f"{dir}/final.pt")


if __name__ == "__main__":
    train()

