import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, device, tokenizer, source_sentences, target_sentences):
        
        self.device = device
        self.tokenizer = tokenizer
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_sentences)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = self.source_sentences[index]
        target_text = self.target_sentences[index]

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=15,
            truncation=True,
            padding = 'max_length',
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=15,
            padding = 'max_length',
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return (
            source_ids.to(self.device, dtype=torch.long).clone().detach(),
            source_mask.to(self.device, dtype=torch.long).clone().detach(),
            target_ids.to(self.device, dtype=torch.long).clone().detach(),
        )
            
            
            
