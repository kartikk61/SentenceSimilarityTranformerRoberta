"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
from datetime import datetime
import torch



# Read the dataset
model_name = 'nli-distilroberta-base-v2'
# model_name='bert-base-nli-mean-tokens'
train_batch_size = 5
num_epochs = 4
model_save_path = 'output/continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)


import pandas as pd

data=pd.read_csv('train_split.csv')
# Get the sentences and similarity scores from the respective columns
sentences_a = data['text1'].tolist()
sentences_b = data['text2'].tolist()
similarity_scores = data['cosineSimilarity'].tolist()

# Create InputExample objects
train_samples = []
for i in range(len(sentences_a)):
    example = InputExample( texts=[sentences_a[i],sentences_b[i]], label=similarity_scores[i])
    train_samples.append(example)




train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

from sentence_transformers import evaluation
data_valid=pd.read_csv('valid_split.csv')
sentences1 = data_valid['text1'].tolist()
sentences2 = data_valid['text2'].tolist()
scores = data_valid['cosineSimilarity'].tolist()


evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
# logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance 
#
##############################################################################

# output_dir = '/home/kartikkitukale/Desktop'  # Replace with the desired path to save the model
# torch.save(model, output_dir)
# torch.save(model.state_dict(), output_dir)
model = SentenceTransformer(model_save_path)
test_data=pd.read_csv('test_split.csv')
# Get the sentences and similarity scores from the respective columns
sentencesa= test_data['text1'].tolist()
sentencesb = test_data['text2'].tolist()
similarityscores = test_data['cosineSimilarity'].tolist()

test_samples = []
for i in range(len(sentencesa)):
    example = InputExample(texts=[sentencesa[i],sentencesb[i]], label=similarityscores[i])
    test_samples.append(example)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)

