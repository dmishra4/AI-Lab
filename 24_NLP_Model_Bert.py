import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD EXCEL FILE
# =========================

file_path = r"bert_log_classification_dataset_2100.xlsx"

df = pd.read_excel(file_path)

print("\n===== ORIGINAL DATA =====")
print(df.head())

# =========================
# CHECK COLUMN NAMES
# =========================

print("\n===== COLUMN NAMES =====")
print(df.columns)

# Replace with your actual column names
text_col = "Log_Message"
label_col = "Category"

# =========================
# REMOVE MISSING VALUES
# =========================

df = df.dropna(subset=[text_col, label_col])

print("\n===== AFTER REMOVING NULLS =====")
print(df.shape)

# =========================
# TEXT CLEANING FUNCTION
# =========================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning
df["Cleaned_Text"] = df[text_col].astype(str).apply(clean_text)

print("\n===== CLEANED DATA =====")
print(df[["Cleaned_Text"]].head())

# =========================
# LABEL ENCODING
# =========================

label_encoder = LabelEncoder()

df["Label"] = label_encoder.fit_transform(df[label_col])

print("\n####Labels##### :: ",label_encoder.classes_)

print("\n===== LABEL MAPPING =====")

for index, label in enumerate(label_encoder.classes_):
    print(f"{label} --> {index}")

#=========================
# FEATURES & TARGET
# =========================

X = df["Cleaned_Text"]
y = df["Label"]

print("\n===== FEATURES SIZE =====")
print(len(X))

print("\n===== LABEL SIZE =====")
print(len(y))

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# OUTPUT
# =========================

print("\n===== TRAIN TEST SPLIT =====")

print("Training Data :", len(X_train))
print("Testing Data  :", len(X_test))

print("\n===== SAMPLE TRAIN DATA =====")

for i in range(5):
    print("\nTEXT :", X_train.iloc[i])
    print("LABEL:", y_train.iloc[i])

# # =========================
# # BERT TOKENIZATION
# # =========================
#
# print("\n===== BERT TOKENIZATION =====")
#
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
# # # Check first 5 training sentences
# # for i in range(5):
# #
# #     # Take one sentence
# #     text = X_train.iloc[i]
# #
# #     print("\nTEXT :", text)
# #
# #     # Convert sentence into tokens
# #     tokens = tokenizer.tokenize(text)
# #
# #     print("TOKENS:")
# #     print(tokens)
# #
# #     # Convert tokens into token IDs
# #     token_ids = tokenizer.convert_tokens_to_ids(tokens)
# #
# #     print("TOKEN IDS:")
# #     print(token_ids)
#
# # =========================
# # BERT ENCODING  ( Single Data )
# # =========================
# #
# # print("\n===== BERT ENCODING =====")
# #
# # # Take first training sentence
# # text = X_train.iloc[0]
# #
# # print("\nORIGINAL TEXT:")
# # print(text)
# #
# # # Full BERT encoding
# # encoded = tokenizer(
# #     text,
# #     padding="max_length",
# #     truncation=True,
# #     max_length=10
# # )
# #
# # print("\nENCODED OUTPUT:")
# # print(encoded)
#
# =========================
#  BERT ENCODING ( FULL DATASET )
# =========================

print("\n===== FULL DATASET ENCODING =====")

# Encode training data
train_encodings = tokenizer(
    list(X_train),
    padding=True,
    truncation=True,
    max_length=20
)

# Encode testing data
test_encodings = tokenizer(
    list(X_test),
    padding=True,
    truncation=True,
    max_length=20
)

print("\nTRAIN INPUT IDS SAMPLE:")
print(train_encodings["input_ids"][0])

print("\nTRAIN ATTENTION MASK SAMPLE:")
print(train_encodings["attention_mask"][0])

# =========================
# PYTORCH DATASET
# =========================

class LogDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {}

        # Add encoded values
        for key, value in self.encodings.items():

            item[key] = torch.tensor(value[idx])

        # Add label
        item["labels"] = torch.tensor(self.labels.iloc[idx])

        return item

    def __len__(self):

        return len(self.labels)

    # =========================
    # CREATE DATASET OBJECTS
    # =========================

    # Create training dataset
train_dataset = LogDataset(train_encodings, y_train)

    # Create testing dataset
test_dataset = LogDataset(test_encodings, y_test)

print("\n===== DATASET CREATED =====")

print("Training Dataset Size :", len(train_dataset))
print("Testing Dataset Size  :", len(test_dataset))

print("\n===== ONE TRAINING DATASET ITEM =====")

print(train_dataset[0])

# # =========================
# # LOAD PRETRAINED BERT MODEL
# # =========================
#
print("\n===== LOADING BERT MODEL =====")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

print("\nBert Model :::", model)

# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=1,

    per_device_train_batch_size=8,

    logging_steps=10
)

# # =========================
# # TRAINER
# # =========================
#
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset
)

# =========================
# START TRAINING
# =========================

print("\n===== TRAINING STARTED =====")

trainer.train()

print("\n===== TRAINING COMPLETED =====")

# =========================
# PREDICTION
# =========================

print("\n===== PREDICTION =====")

# New custom log message
new_text = "database connection timeout"

print("\nINPUT TEXT:")
print(new_text)

# Encode new text
inputs = tokenizer(

    new_text,

    return_tensors="pt",

    padding=True,

    truncation=True,

    max_length=20
)

# Disable gradient calculation
with torch.no_grad():

    outputs = model(**inputs)

# Get prediction
predicted_label = torch.argmax(outputs.logits, dim=1).item()

print("\nPREDICTED LABEL ID:")
print(predicted_label)

# Convert label ID back to original category
predicted_category = label_encoder.inverse_transform([predicted_label])

print("\nPREDICTED CATEGORY:")
print(predicted_category[0])

# =========================
# MODEL EVALUATION
# =========================

print("\n===== MODEL EVALUATION =====")

# Get predictions on test dataset
predictions = trainer.predict(test_dataset)

# Predicted label IDs
y_pred = predictions.predictions.argmax(axis=1)

# Actual labels
y_true = y_test.tolist()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\nACCURACY:")
print(accuracy)

# Detailed report
print("\nCLASSIFICATION REPORT:")

print(classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_
))

# =========================
# SAVE MODEL
# =========================

print("\n===== SAVING MODEL =====")

# Save trained model
model.save_pretrained("./saved_bert_model", safe_serialization= False)

# Save tokenizer
tokenizer.save_pretrained("./saved_bert_model")

print("\nMODEL SAVED SUCCESSFULLY!")