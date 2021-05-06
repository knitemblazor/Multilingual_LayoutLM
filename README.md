# Multilingual_LayoutLM 

This project has been derived from microsoft's LayoutLM project with dependency for transformers removed.
It also includes support for 140 languages.This is a completed project  for training and prediction  of multilingual documents as there are limitations on labelled dataset kindly prepare data for your respective languages.I have currently tested it for hindi, malayalam, english combinations. I have released the training flow and model accordingly for these languages  that have been trained on adhaar dataset

## Model Link
path and config file for multilingual bert model for producing embeddings  \
https://drive.google.com/drive/folders/1t5Ktz94YTSrE_JHdrfiPc4Moi-K4GxHz?usp=sharing

## pretraining flow
To do


## Training flow
Training flow is in the train directory \
Do alter and go through the parameters in the config.yml inside train directory to suit your requirements. 

## Steps
1. clone the repository
2. run pip install -r requirements.txt
3.  **To train** go to train folder and run python train_eval.py after making changes in the config file \
   you should also download the pretrained model from the given link and place it in the folder models \
   similarly you should prepare the data in the format as in folder annotated_adhaar_data \
4. **To predict** alter the config file outside the train folder and ***run python parser.py*** with the image path \
   after putting it in the parser.py file  
    


