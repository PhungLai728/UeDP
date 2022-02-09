# UeDP (User and Entity Differential Privacy Preservation in Natural Language Models)

## Software Requirements
Python 3.7 and Pytorch 1.5.1 are used for the current codebase. 
We recommend you to create an environment and install necessary packages (e.g., numpy, pickle, pandas, codecs, etc.)
We ran our experiment on NVIDIA GeForce Titan Xp (GPU) or Intel Xeon E5-2637 v4 @ 3.50GHz (CPU), Linux (Ubuntu 16.04), CUDA 10.0, CuDNN 7.6.0.
For the BERT and GPT-2 models, we use transformers version 4.10.

## Experiments
The repository comes with instructions to reproduce the results in the paper or to train the model with your dataset:

To run the default settings and reproduce the results: 
+ For next word prediction: Run `python3 nwp_gpt2_xx.py` for using the GPT-2 model or `python3 main_lm_xx.py` for using the AWD-LSTM model, where `xx` relies on the task and data you want to run, such as: 
`python3 nwp_conll_gpt2.py` is for running Noiseless GPT-2 model with the AG dataset,
`python3 nwp_conll_gpt2_UeDP.py` is for running UeDP with the GPT-2 model with the AG dataset,
`python3 main_lm_conll_UeDP.py` is for running UeDP with the CONLL-2003 dataset,
`python3 main_lm_conll_UserDP.py` is for running User-level DP with the CONLL-2003 dataset,
`python3 main_lm_conll_DeInd.py` is for running De-Identification with the CONLL-2003 dataset,
`python3 main_lm_conll_Noiseless.py` is for running Noiseless AWD-LSTM with the CONLL-2003 dataset,
Similar with the AG dataset, just replacing `conll` by `ag` and then run the task you want.

+ For text classification: Run `python3 text_classification_xx.py` where `xx` relies on the task, model, and data you want to run, such as: 
`python3 text_classification_ag_bert_UeDP.py` is for running UeDP with BERT,
`python3 text_classification_ag_bert_UserDP.py` is for running User-level DP with BERT,
`python3 text_classification_ag_awdlstm_UeDP.py` is for running UeDP with AWD-LSTM,
`python3 text_classification_ag_awdlstm_UserDP.py` is for running User-level DP with AWD-LSTM,
`python3 text_classification_ag_awdlstm_DeInd.py` is for running De-Identification with AWD-LSTM,
`python3 text_classification_ag_awdlstm_Noiseless.py` is for running Noiseless AWD-LSTM.

+ There are several hyper-parameters that you can tune to achieve a good result. Please refer to the parser list in the codes. For example, clipping bound (--clip), entity type (--ent), learning rate (--lr), number of user per iteration (--nu), etc.
By adding these parsers after your command, then it works, for example `python3 main_lm_conll_ueDP.py --clip 0.1 --ent loc`. 

+ Note: Due to the privacy requirements of SEC data, this repository only provides data and code for CONLL-2003 and AG datasets. 

+ For the pretrained Glove, please download `glove.6B.100d.txt` from the following link: https://nlp.stanford.edu/projects/glove/

To customize the code with your data:
+ First, you need to get a set of sensitive entities. Please refer to the following tool-kits where you can customize the set: 

(1) Spacy: https://spacy.io/api/data-formats#named-entities (Demo: https://explosion.ai/demos/displacy-ent)

(2) Stanza: https://spacy.io/universe/project/spacy-stanza

(3) Microsoft Presidio: https://microsoft.github.io/presidio/ (Demo: https://presidio-demo.azurewebsites.net/)

+ After that, consider the sensitive entity set as one of the inputs in our UeDP mechanism.











