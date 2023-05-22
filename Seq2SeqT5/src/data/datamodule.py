from torch.utils.data import DataLoader

from data.custom_dataset import CustomDataset
from transformers import T5Tokenizer
import sentencepiece as spm
from data.utils import TextUtils, short_text_filter_function
import pandas as pd
import pickle
import os

class DataManager:
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        
        # read and filter senrences
        pairs = pd.read_csv(self.config["filename"], sep='\t', header=None, names=['src', 'tgt'], usecols=[0, 1]).sample(frac=1).values
        source_sentences,target_sentences = [], []
        unique_sources = set()
        for pair in pairs:
            source, target = pair[0], pair[1]
            if short_text_filter_function(pair, self.config['max_length'], None) and source not in unique_sources:
                source_sentences.append(source)
                target_sentences.append(target)
                unique_sources.add(source)

        # train valid split
        train_size = int(len(source_sentences)*self.config["train_size"])
        source_train_sentences, source_val_sentences = source_sentences[:train_size], source_sentences[train_size:]
        target_train_sentences, target_val_sentences = target_sentences[:train_size], target_sentences[train_size:]

        # init T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-efficient-tiny")
        
        
        # add new tokens
        with open('./data/train_sentences.txt', 'w') as file:
            print('\n'.join(source_train_sentences + target_train_sentences), file=file)
        spm.SentencePieceTrainer.train(input='./data/train_sentences.txt', model_prefix='./data/m', model_type='bpe', vocab_size=15_000)
        with open('./data/m.vocab', 'r') as file:
            new_tokens = file.readlines()
        new_tokens = [x.split('\t')[0] for x in new_tokens]
        print('orig', len(self.tokenizer.get_vocab()))
        print('from source', len(new_tokens))
        print('added', self.tokenizer.add_tokens(new_tokens))
        
        # init dataloaders
        train_dataset = CustomDataset(self.config['device'], self.tokenizer, source_train_sentences, target_train_sentences)
        val_dataset = CustomDataset(self.config['device'], self.tokenizer, source_val_sentences, target_val_sentences)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.config["batch_size"], drop_last=True)
        val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=self.config["batch_size"], drop_last=True)
        
        return train_dataloader, val_dataloader
