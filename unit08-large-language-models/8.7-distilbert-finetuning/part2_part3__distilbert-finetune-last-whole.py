# Finetuning a DistilBERT Classifier in Lightning

#Here, we are finetuning the output layers of a pretrained transformer:

#![](figures/finetuning-i.png)

# pip install transformers

# pip install datasets

# pip install lightning

#%load_ext watermark
#%watermark --conda -p torch,transformers,datasets,lightning

# 1 Loading the dataset into DataFrames

# pip install datasets

import os.path as op

from datasets import load_dataset

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import CountVectorizer

from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, partition_dataset
from local_dataset_utilities import IMDBDataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import lightning as L
import torch
import torchmetrics
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# 3 Set Up DataLoaders

from torch.utils.data import DataLoader, Dataset

class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows

finetuneLast = False
def main():
    torch.set_float32_matmul_precision('medium')    
    if not  op.exists("train.csv") or not  op.exists("val.csv") or not  op.exists("test.csv"):
        download_dataset()

        df = load_dataset_into_to_dataframe()
        partition_dataset(df)

    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("val.csv")
    df_test = pd.read_csv("test.csv")

    # 2 Tokenization and Numericalization

    #**Load the dataset via `load_dataset`**

    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
    )

    print(imdb_dataset)

    #**Tokenize the dataset**

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)

    del imdb_dataset

    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

          
    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True, 
        num_workers=4
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=4
    )

    # 4 Initializing DistilBERT



    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)

    #**Freeze all layers**
    if finetuneLast :
        for param in model.parameters():
            param.requires_grad = False

        #**Unfreeze last layer**


        for param in model.pre_classifier.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

    model


    class LightningModel(L.LightningModule):
        def __init__(self, model, learning_rate=5e-5):
            super().__init__()

            self.learning_rate = learning_rate
            self.model = model

            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

        def forward(self, input_ids, attention_mask, labels):
            return self.model(input_ids, attention_mask=attention_mask, labels=labels)
            
        def training_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["label"])        
            self.log("train_loss", outputs["loss"])
            return outputs["loss"]  # this is passed to the optimizer for training

        def validation_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["label"])        
            self.log("val_loss", outputs["loss"], prog_bar=True)
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.val_acc(predicted_labels, batch["label"])
            self.log("val_acc", self.val_acc, prog_bar=True)
            
        def test_step(self, batch, batch_idx):
            outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["label"])        
            
            logits = outputs["logits"]
            predicted_labels = torch.argmax(logits, 1)
            self.test_acc(predicted_labels, batch["label"])
            self.log("accuracy", self.test_acc, prog_bar=True)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer
        

    lightning_model = LightningModel(model)

    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="val_acc"
        )  # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name="my-model")

    trainer = L.Trainer(
        max_epochs=3,
        callbacks=callbacks,
        accelerator="gpu",
        precision="16-mixed",
        devices=[0],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best")
    trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best")
    trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")

if __name__ == '__main__':
    main()
    ...
