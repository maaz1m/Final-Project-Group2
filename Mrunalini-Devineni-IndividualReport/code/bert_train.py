from datasets import load_dataset

emotion_raw = load_dataset("emotion")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


emotion_tokenized = emotion_raw.map(tokenize_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=6)

from datasets import load_metric

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=emotion_tokenized["train"],
    eval_dataset=emotion_tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

predictions = trainer.predict(emotion_tokenized['test'])

import numpy as np
predicted_labels = np.argmax(predictions.predictions, axis=-1)
target_labels = predictions.label_ids

from sklearn.metrics import accuracy_score, f1_score

f1_score(target_labels, predicted_labels, average='macro')
accuracy_score(target_labels, predicted_labels)