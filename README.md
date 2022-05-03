# Final-Project-Group2
 NLP Final Project Group 2

## Description
In our project, we aim to analyze the impact of emotion of cryptocurrency market players on the price of Bitcoin using the sentiment classification abilities of transformer based natural language processing techniques. 

## Goals
* Develop a dataset of Reddit comments from the r/cryptocurrency subreddit
* Fine tune a transformer model on an emotion classification dataset
* Repurpose the model to predict the emotion exhibited by Reddit comments
* Investigate trends between certain emotions and the price of Bitcoin

## Requirements

There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.
* `pandas`
* `numpy`
* `seaborn`
* `matplotlib`
* `transformers`
* `sklearn`
* `pytorch`

## Source Code

### reddit_pull.py
Loads reddit 18million comments from r/cryptocurrency and saves as `cryptocurrency_comments_raw.csv`

### preprocessing.py
Cleans comments dataset and saves random sample of 500k to `cryptocurrency_comments_cleaned_subset.csv`

### bert_train.py
Trains BERT model on emotions dataset and saved checkpoint

### bert_infer.py
Loads BERT model from checkpoint and generates predictions on reddit comments and saves as `cryptocurrency_comments_cleaned_subset_emotion.csv`

### prices_pull.py
Loads prices of Bitcoin from last year and saves as `cryptocurrency_prices.csv`

### results.py
Loads comments, predicted emotion and BTC prices to draw line plots for comparison.

All datasets may be downloaded from this [link](https://pern-my.sharepoint.com/:f:/g/personal/20100068_lums_edu_pk/Eq9-txl6vSdDpGqg4L1BIn0BeDAf5C8Co90l9gGC3vUfPA?e=gMQ6qB)
