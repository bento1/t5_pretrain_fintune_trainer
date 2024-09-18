
import pytorch_lightning as pl
import random
import torch
import numpy as np
import logging
import os 

from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM

)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FintuneModel(pl.LightningModule):
    args={}
    train_step_outputs=[]
    validation_step_outputs=[]
    def __init__(self,args):
        super(FintuneModel, self).__init__()
        self.args=args
        self.train_step_outputs=[]
        self.validation_step_outputs=[]
        self.automatic_optimization = False
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path) 
        #   self.model.dropout_rate=0.2
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        if self.args.freeze_embeds:
            self.freeze_embeds()
        if self.args.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())
        

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    
    def is_logger(self):
        # return self.trainer.proc_rank <= 0 #old
        return self.trainer.global_rank <= 0
    
    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        decoder_attention_mask=decoder_attention_mask)
        return outputs.loss, outputs.logits
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        
        return loss


    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        loss = self._step(batch)
        print(loss)
        self.train_step_outputs.append(loss.cpu())
        # Insert these lines:
        self.manual_backward(loss)
        
        optimizer = self.optimizers()
        # scheduler = self.lr_schedulers()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}



    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.model.eval()
        with torch.no_grad():
            loss = self._step(batch)
        self.validation_step_outputs.append(loss.cpu())
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss, output = self(input_ids=input_ids, attention_mask=attention_mask)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0,
                num_training_steps=self.args.num_train_epochs*self.args.n_train)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))
