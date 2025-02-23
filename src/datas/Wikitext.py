import transformers
import datasets
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(
            self                           ,
            split           : str          ,
            dataset_size    : int   = None ,
            max_length      : int   = 128  ,
            map_batch_size  : int   = 2000 ,
            mlm_probability : float = .25  ,
            map_proc        : int   = 4    ,
            seed            : int   = 42   ,
        ):

        self.split      = split
        self.max_length = max_length

        self.raw_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split)
        self.tokenizer   = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.raw_dataset = self.raw_dataset.filter(lambda x: len(x['text']) > 0)
        self.raw_dataset = self.raw_dataset.shuffle(seed=seed)

        if dataset_size is not None:
            self.raw_dataset = self.raw_dataset.select(range(min(len(self.raw_dataset),dataset_size)))

        self.tok_dataset = self.raw_dataset.map(
            self.tokenize, 
            batched        = True           ,
            num_proc       = map_proc       ,
            batch_size     = map_batch_size ,
            remove_columns = ['text']
        )

        self.collator = transformers.DataCollatorForLanguageModeling(
            tokenizer       = self.tokenizer  ,
            mlm_probability = mlm_probability ,
            mlm             = True            ,
        )

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], truncation=True, padding='max_length', max_length=self.max_length)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        return self.tok_dataset[idx]

    def collate_fn(self, data):
        return self.collator(data)
