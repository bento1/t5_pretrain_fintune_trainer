from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch




class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(self, 
                train_df, 
                val_df,
                test_df,
                batch_size,
                tokenizer,
                text_len,
                summary_len):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.summary_len = summary_len
    
    def setup(self, stage=None):
        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_len,
            self.summary_len)
        
        self.val_dataset = NewsSummaryDataset(
            self.val_df,
            self.tokenizer,
            self.text_len,
            self.summary_len)
        
        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_len,
            self.summary_len
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            #num_workers=4
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            #num_workers=4
            )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            #num_workers=4
        )

class NewsSummaryDataset(Dataset):
    def __init__(self, df, tokenizer, text_len, headline_len):
        self.df = df
        self.headlines = self.df["headlines"]
        self.text = self.df["text"]
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.headline_len = headline_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        # T5 transformers performs different tasks by prepending the particular prefix to the input text.
        text = "summarize:" + str(self.text[idx])                # In order to avoid dtype mismatch, as T5 is text-to-text transformer, the datatype must be string
        headline = str(self.headlines[idx])

        text_tokenizer = self.tokenizer(text, max_length=self.text_len, padding="max_length",
                                                        truncation=True, add_special_tokens=True)
        headline_tokenizer = self.tokenizer(headline, max_length=self.headline_len, padding="max_length",
                                                        truncation=True, add_special_tokens=True)
        return {
            "input_ids": torch.tensor(text_tokenizer["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(text_tokenizer["attention_mask"], dtype=torch.long),
            "target_ids": torch.tensor(headline_tokenizer["input_ids"], dtype=torch.long),
            "target_mask": torch.tensor(headline_tokenizer["attention_mask"], dtype=torch.long)
        }