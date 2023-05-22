import sys
sys.path.append('./src/')
sys.path.append('./src/data/')
sys.path.append('./src/models/')

import torch
import torch.nn as nn
import yaml
from models import trainer
from data.datamodule import DataManager
from txt_logger import TXTLogger
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor
from torchtext.data.metrics import bleu_score

BATCH_SIZE = 128
RANDOM_STATE = 456
MAX_LEN = 15
LEARNING_RATE = 0.001
SHEDULER_STEP_SIZE = 1000
EPOCH_NUM = 5000
TRY_ONE_BATCH = False
PREFIX_FILTER = None
FILENAME = '../data/rus.txt'
TRAIN_SIZE = 0.8

class Seq2Seq_T5_Trainer(nn.Module):
    def __init__(self, model, optimizer, scheduler, tokenizer):
        super(Seq2Seq_T5_Trainer, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        
    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2]).loss
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss
        
    def validation_step(self, batch):
        self.model.eval()
        with torch.no_grad():

            loss = self.model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2]).loss
        
        return loss

    def forward(self, batch):
        
        generated_ids = self.model.generate(
            input_ids = batch[0],
            attention_mask = batch[1], 
            max_length=15, 
            num_beams=2,
            repetition_penalty=2.5, 
            length_penalty=1.0, 
            early_stopping=True
          )
        preds =  [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in batch[2]     ]
          
        return preds, target
            
    def eval_bleu(self, predicted, actual):
        words_predicted = [x.split() for x in predicted]
        words_actual = [[x.split()] for x in actual]
        score = bleu_score(words_predicted, words_actual, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
        return score, actual, predicted
    
if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config = dict()
    config['batch_size'] = BATCH_SIZE
    config['prefix_filter'] = PREFIX_FILTER
    config['max_length'] = MAX_LEN
    config['epoch_num'] = EPOCH_NUM
    config['try_one_batch'] = TRY_ONE_BATCH
    config['learning_rate'] = LEARNING_RATE
    config['device'] = DEVICE
    config['filename'] = FILENAME
    config['train_size'] = TRAIN_SIZE

    dm = DataManager(config)
    train_dataloader, dev_dataloader = dm.prepare_data()

    VOCAB_SIZE = len(dm.tokenizer)
    config['vocab_size'] = VOCAB_SIZE
    
    model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny")
    model.resize_token_embeddings(VOCAB_SIZE)
    model.to(DEVICE)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adafactor(model.parameters(), lr=config['learning_rate'], relative_step=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SHEDULER_STEP_SIZE, gamma=0.99)
    
    model_trainer = Seq2Seq_T5_Trainer(model, optimizer, scheduler, dm.tokenizer)    
    logger = TXTLogger('training_logs')
   
    trainer_cls = trainer.Trainer(model=model_trainer, config=config, logger=logger)

    if config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)