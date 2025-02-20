from typing import Tuple, Optional
from utils import Any
from valid import valid
import models
import utils
import torch
import datas
import click


@click.command()
@click.option("--seed"            , "seed"            , type=int                           , default=42     , help="Random seed"                       )
@click.option("--test-batch-size" , "test_batch_size" , type=int                           , default=256    , help="test batch size"                   )
@click.option("--device"          , "device"          , type=str                           , default="cuda" , help="Device to train on"                )
@click.option("--compile"         , "compile"         , type=bool                          , default=False  , help="Compile the model"                 )
@click.option("--dir"             , "dir"             , type=click.Path()                  , default="data" , help="Directory to save data"            )
@click.option("--restore"         , "restore"         , type=click.Path()                  , default=None   , help="Path to restore"                   )
@click.option("--num-workers"     , "num_workers"     , type=int                           , default=0      , help="Number of workers for dataloader"  )
@click.option("--arch"            , "arch"            , type=(str, Any(int,float,bool,str)), default=[]     , help="Model architecture", multiple=True )
def test(
        seed            : int                                      = 42     ,
        test_batch_size : int                                      = 256    ,
        compile         : bool                                     = False  ,
        dir             : str                                      = "data" ,
        device          : str                                      = "cuda" ,
        num_workers     : int                                      = 0      ,
        restore         : Optional[str]                            = None   ,
        arch            : Optional[Tuple[str, int|float|bool|str]] = None   ,
    ):

    torch.set_float32_matmul_precision('high')

    # seed everything 
    utils.seed_all(seed)

    # instantiate the model
    compiler = torch.compile if compile else lambda x: x
    model    = compiler(models.Bert(**dict(arch)).to(device))
    if restore: model.load(restore)

    # stuff for training
    testloss   = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    testlogger = utils.get_logger(f"{dir}/test.jsonl")

    # instantiate the dataset
    test_loader = torch.utils.data.DataLoader(
        dataset := datas.WikitextDataset(split="test"),
        batch_size     = test_batch_size     ,
        collate_fn     = dataset.collate_fn  ,
        num_workers    = num_workers         ,
    )

    print(valid(
        model        = model        ,
        loss         = testloss     ,
        loader       = test_loader  ,
        logger       = testlogger   ,
        device       = device       ,
        progress_bar = None         ,
        epoch        = -1           ,
    ))

if __name__ == "__main__":
    test()


