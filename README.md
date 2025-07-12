# Intent Based Counter Speech Generation
> The aim of the project is to train a Large Language Model, which can reply to the hatepseech based on the given intent. We hypothesize that by explicity providing the target intent as part of input, the model must be able to generate more diverse and targeted counterspeech. 

## Abstract
Mitigating online hate speech is crucial, and intent-specific counterspeech offers a promising approach. This research focuses on
developing models for this task using the IntentCONANv2 dataset, which comprises 13K hate speech-counterspeech pairs conditioned on four intents: informative, denouncing, question, and positive. We investigate the fine-tuning of two pre-trained Transformer architectures, T5
(a sequence-to-sequence model) and GPT-2 (a decoder-only model), to generate counterspeech given a input hate speech and conditioned on specified csType (intent). Multiple Experiments has been performed on T5 Model with multiple variations. Model performance is evaluted using BLEU, ROUGE, and BERTScore.

## Dataset Used
Intent Conan V2
Dataset Link: https://huggingface.co/datasets/Aswini123/IntentCONANv2(Private Dataset)

## Highlights
- Finetuned on two seperate models with various techniques. One model was T5(seq-2-seq) and GPT2(Decoder-only Model).
- Other than baselines, prompt augmentation and self contranstive learning methods were applied on T5 Model.
- Dataset: IntentConanV2
- Achieved: Rouge Score: 22.521 Bleu Score: 13.650 Bert score(F1): 86.239. it was achieved on one of the baselines, and thus it is better than baseline model(T5 Model).

## Methodology
### Dataset
We use the IntentCONANv2 dataset, which contains ~13,000 examples of hate speech paired with human-written counterspeech. Each sample also includes an intent label, which tells us the purpose of the counterspeech:

- Informative (fact-based),
- Denouncing (condemns the hate),
- Question (asks thoughtful questions),
- Positive (offers support or positivity).

### Model & Training

We started by comparing T5 (encoder-decoder) and GPT-2 (decoder-only) using prompt-based inference. T5 showed better results for generating intent-specific counterspeech, so we chose it for all further experiments.

Our main experiments include:

- LoRA (Low-Rank Adaptation): Fine-tuned T5 using LoRA, which updates fewer parameters to reduce compute cost.

- Two-Stage Fine-Tuning:
        1: Train on general counterspeech (no intent).
        2: Fine-tune again with intent-specific prompts.

- Prompt Augmentation: Improve model robustness by training with various templates and adding small changes (typos, synonyms) to prompts.
- Intent Contrastive Self-Correction (ICS): Trained the model with both correct and incorrect intent labels. A special loss function helps it learn to generate responses that strongly match the correct intent.

(More Details about the methodology can be accessed on the report given in the docs folder.)

## Folder Structure
Overview:
- requirements.txt: All required libraries to run the library.
- src/ : This folder contains all the python scripts used to run the experiments.
- doc/ : Contains the report and presentation for the project
- datasets/: Contains the actual dataset from huggingface which has been used for the finetuning the model.

## Set up the Environment
conda env create -n t5-lora
conda activate t5-lora
pip install -r requirements.txt

## 🧪 Running the Code
After setting up the environemnt, set up the hugging face key(I have removed mine from the code). You need to add a line for the key verification to access the dataset. After making the changes, you can just run the python file.
python src/<file_name>.py

## References

- Benesch, S., et al. (2016). Dangerous Speech: A Practical Guide. Dangerous SpeechProject.
- Chung,M., et al. (2021). Towards a Taxonomy of Counterspeech for Combating Online Hate Speech.Proceedings of the International AAAI Conference on Web and Social Media(ICWSM).
- Aswini, A. Intent-conditioned and Nontoxic Counterspeech Generation using Multi Task Instruction Tuning with RLAIF.

## Developer and License 
This project is a part of the academic course project submitted at the end of the course "Deep Learning in Mechanics - ELL884", taught by the Prof. Tanmoy Chakraborty.

Student Name: Somesh Agrawal

<!-- Team Name: Lost All Hopes -->

Github Link: https://github.com/somesh2002/counter-speech-intent-conan-v2

This code given in the file can be tweaked and used as per your convience, but you are not allowed to access the dataset, and it is a property of the actual authors and you can this dataset from the Hugging Face or may contact the actual administrator.

