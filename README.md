## Overview
This is a practice project that investigated pretraining and finetuning Transformer self-attention building blocks. This project involves training a Transformer model to attempt to answer simple questions of the form “_Where was person [x] born?_” – without providing any input text from which to draw the answer. The model was pretrained on Wikipedia text that contains world knowledge, then finetune the model on the knowledge intensive task (birth place dataset), enabling the model to access some knowledge learned during funetuning.

In this practice, I used the span corruption technique from the T5 paper, randomly selects spans of text in a document and replaces with unique tokens(noises). Model then takes these noises and output a pattern of each unique sentinal followed by the noises. Additionally the model is implemented with a more efficient variant of attention PerceiverAR(a simpler version). It reduced the sequence length of the input to self-attention for the input layer, by allowing the input to be projected to a smaller subspace and then project back to the original input length in the last layer.

## Bash Commands
Below are bash commands to pretrain the model, finetune it, and make predictions on the dev and test sets. Note that the pretraining process will take approximately 2 hours on GDP.
```
# Pretrain the model
python src/run.py pretrain perceiver wiki.txt --bottleneck_dim 64 \
       --pretrain_lr 6e-3 --writing_params_path perceiver.pretrain.params

# Finetune the model
python src/run.py finetune perceiver wiki.txt --bottleneck_dim 64 \
       --reading_params_path perceiver.pretrain.params \
       --writing_params_path perceiver.finetune.params \
       --finetune_corpus_path birth_places_train.tsv

# Evaluate on the dev set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 \
       --reading_params_path perceiver.finetune.params \
       --eval_corpus_path birth_dev.tsv \
       --outputs_path perceiver.pretrain.dev.predictions

# Evaluate on the test set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 \ --reading_params_path perceiver.finetune.params \ --eval_corpus_path birth_test_inputs.tsv \
--outputs_path perceiver.pretrain.test.predictions
```

## Reference
The code is a fork of Andrej Karpathy’s minGPT, and the original structure is from CS224N class materials.