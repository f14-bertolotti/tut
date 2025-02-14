import transformers
import datasets
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(
            self                          ,
            split          : str          ,
            max_length     : int = 128    ,
            map_batch_size : int = 2000   ,
            map_proc       : int = 4      ,
            device         : str = "cuda" ,
        ):

        self.split      = split
        self.device     = device
        self.max_length = max_length

        self.raw_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split)
        self.tokenizer   = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

        self.tok_dataset = self.raw_dataset.map(
            self.tokenize, 
            batched        = True           ,
            num_proc       = map_proc       ,
            batch_size     = map_batch_size ,
            remove_columns = ['text']
        )

        self.collator = transformers.DataCollatorForLanguageModeling(
            tokenizer       = self.tokenizer ,
            mlm_probability = 0.15           ,
            mlm             = True           ,
        )

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], truncation=True, padding='max_length', max_length=self.max_length)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        return self.tok_dataset[idx]

    def collate_fn(self, data):
        return self.collator(data)
