# Natural Language Inference (NLI) Kaggle Competition

This repository contains code and experiments related to the **Contradictory, My Dear Watson** [Kaggle competition](https://www.kaggle.com/competitions/contradictory-my-dear-watson). My main motivation for this project was to play around with my new [RTX 3090 PC build](https://pcpartpicker.com/user/arnePannemans/saved/MgtPzy) and experiment with state-of-the-art (smaller) large language models to see how fine-tuning can improve their performance for the task.


## 📖 Background

The **Contradictory, My Dear Watson** competition challenges participants to build a Natural Language Inference (NLI) model capable of determining the relationship between pairs of sentences across 15 different languages. The possible relationships are:

- **Entailment**
- **Neutral**
- **Contradiction**

For more details about the competition, [check out the official page here](https://www.kaggle.com/competitions/contradictory-my-dear-watson).

## 🚀 Project Overview

This project explores various transformer-based models for the NLI task, testing out both out-of-the-box performance and performance after fine-tuning.

## 🧰 Setup Instructions

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/ArnePannemans/Kaggle_NLI.git
    cd Kaggle_NLI
    ```

2. **Create and activate a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```

3. **Install the required packages**

    ```bash
    pip install -r requirements.txt
    ```

## 📝 Experiments and Results

Below are the results from initial experiments using Phi-3 models:

1. **Phi-3 Mini 4k Instruct (Out-of-the-Box)** : Accuracy: **67.3%**
2. **Phi-3 Mini 4k Instruct (Fine-Tuned)** : Accuracy: **81.1%**
