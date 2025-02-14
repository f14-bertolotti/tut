import models
import utils
import torch
import datas
import click
import tqdm


@click.command()
@click.option("--seed"       , "seed"       , type=int , default=42     , help="Random seed")
@click.option("--batch-size" , "batch_size" , type=int , default=1024   , help="Batch size")
@click.option("--epochs"     , "epochs"     , type=int , default=10     , help="Number of epochs")
@click.option("--device"     , "device"     , type=str , default="cuda" , help="Device to train on")
@click.option("--compile"    , "compile"    , type=bool, default=False  , help="Compile the model")
def train(
        seed        : int = 42     ,
        batch_size  : int = 1024   ,
        epochs      : int = 10     ,
        compile     : bool = False ,
        device      : str = "cuda" ,
    ):

    # seed everything 
    utils.seed_all(seed)

    # instantiate the dataset
    train_loader = torch.utils.data.DataLoader(
        dataset := datas.WikitextDataset(split="train", device=device),
        batch_size  = batch_size         ,
        collate_fn  = dataset.collate_fn ,
        num_workers = 4                  ,
    )

    # instantiate the model
    compiler = torch.compile if compile else lambda x: x
    model    = compiler(models.Bert().to(device))

    # instantiate the loss
    lossfn = torch.nn.CrossEntropyLoss()

    # instantiate the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # start training
    progress_bar = tqdm.tqdm(total=epochs * len(train_loader), desc='Epochs')
    for epoch in range(epochs):
        for batch in train_loader:
            batch  = {k: v.to(device, non_blocking=True) for k,v in batch.items()}
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"      ],
                batch["token_type_ids" ],
                batch["attention_mask" ],
            )
            print(logits.shape)
            #loss   = lossfn(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))


            progress_bar.update(1)

        break

if __name__ == "__main__":
    train()
