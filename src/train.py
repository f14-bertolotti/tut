import typing
import models
import utils
import torch
import datas
import click
import tqdm


@click.command()
@click.option("--seed"       , "seed"       , type=int                 , default=42     , help="Random seed"                       )
@click.option("--batch-size" , "batch_size" , type=int                 , default=1024   , help="Batch size"                        )
@click.option("--epochs"     , "epochs"     , type=int                 , default=10     , help="Number of epochs"                  )
@click.option("--device"     , "device"     , type=str                 , default="cuda" , help="Device to train on"                )
@click.option("--compile"    , "compile"    , type=bool                , default=False  , help="Compile the model"                 )
@click.option("--dir"        , "dir"        , type=click.Path()        , default="data" , help="Directory to save data"            )
@click.option("--etc"        , "etc"        , type=int                 , default=None   , help="epochs to wait before saving"      )
@click.option("--restore"    , "restore"    , type=(click.Path(), int) , default=None   , help="Path to restore from and step num" )
@click.option("--num-workers", "num_workers", type=int                 , default=0      , help="Number of workers for dataloader"  )
def train(
        seed        : int                        = 42     ,
        batch_size  : int                        = 1024   ,
        epochs      : int                        = 10     ,
        compile     : bool                       = False  ,
        dir         : str                        = "data" ,
        etc         : int|None                   = None   ,
        device      : str                        = "cuda" ,
        num_workers : int                        = 0      ,
        restore     : typing.Tuple[str,int]|None = None   ,
    ):

    utils.save_state(f"{dir}/state.json")

    # seed everything 
    utils.seed_all(seed)

    # instantiate the model
    compiler = torch.compile if compile else lambda x: x
    model    = compiler(models.Bert(tie_word_embeddings = True).to(device))

    # stuff for training
    lossfn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    # instantiate the dataset
    train_loader = torch.utils.data.DataLoader(
        dataset := datas.WikitextDataset(split="train"),
        batch_size     = batch_size          ,
        collate_fn     = dataset.collate_fn  ,
        num_workers    = num_workers         ,
    )


    progress_bar = tqdm.tqdm(total=epochs * len(train_loader), desc='steps')
        
    # start training
    for epoch in range(epochs):
        for batch in train_loader:

            if restore is not None and progress_bar.n < restore[1]: 
                progress_bar.update(1)

                if progress_bar.n == restore[1]:
                    model.load(f"{restore[0]}/model{restore[1]}.pth")
                    optim.load_state_dict(torch.load(f"{restore[0]}/optim{restore[1]}.pth"))
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
            loss = lossfn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            acc  = (logits[batch["labels"] != -100].argmax(-1) == batch["labels"][batch["labels"]!=-100]).float().mean()
            loss.backward()
            optim.step()
            progress_bar.set_description(f"e {epoch: <2}, s:{progress_bar.n: <5}, l: {loss.item():5.3f}, a: {acc.item():5.3f}")

            # save checkpoint
            if etc is not None and progress_bar.n % etc == 0:
                model.save(f"{dir}/model{progress_bar.n}.pth")
                torch.save(optim.state_dict(), f"{dir}/optim{progress_bar.n}.pth")
                utils.save_state(f"{dir}/state{progress_bar.n}.json")

            progress_bar.update(1)


if __name__ == "__main__":
    train()

