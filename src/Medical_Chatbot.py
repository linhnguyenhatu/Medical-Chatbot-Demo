# %%
!pip install nltk

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import re
import torchtext.vocab as tvc
from nltk.corpus import stopwords

# %%
nltk.download('wordnet')
nltk.download('stopwords')

# %%
#Import Dataset

df = pd.read_csv('Dataset/train_medical_chatbot.csv', engine='python', on_bad_lines='skip')
df


# %%
#Preprocess dataset
wordnet = WordNetLemmatizer()
stop_words = stopwords.words('english')
def get_rid_of_q(text):
  if text is None:
      return ""  
  return text.replace("Q. ", "")

def clean_text(text):
  text = get_rid_of_q(text)
  text = re.sub('[^a-zA-Z]', ' ', text)
  text = text.lower()
  text = text.split()
  text = [wordnet.lemmatize(word) for word in text][:1000]
  return text

def add_eos(text):
  text = text + ["<eos>"]
  return text
df["Description"] = df["Description"].apply(clean_text)
df["Patient"] = df["Patient"].apply(clean_text)
df["Patient"] = df["Patient"].apply(add_eos)
df["Doctor"] = df["Doctor"].apply(clean_text)
df["Doctor"] = df["Doctor"].apply(add_eos)

df

# %%
#Build/load vocabulary
unk_token = "<unk>"
pad_token = "<pad>"
eos_token = "<eos>"
special_tokens = [unk_token, pad_token, eos_token]
min_freq = 2
whole_text = df["Description"].tolist() + df["Patient"].tolist() + df["Doctor"].tolist()
text_vocabulary = torch.load('Load_Vocabulary/text_vocabulary.pt')
text_vocabulary.set_default_index(text_vocabulary[unk_token])

# %%
torch.save(text_vocabulary, 'Load_Vocabulary/text_vocabulary.pt')

# %%
#Combine description and patient prompt
df["Patient"] = df["Description"] + df["Patient"]

# %%
#Numberize the tokens
def token_to_index(tokens):
  return text_vocabulary.lookup_indices(tokens)

# %%
#Numberizing and padding the dataset

df["Patient"] = df["Patient"].apply(token_to_index)
df["Doctor"] = df["Doctor"].apply(token_to_index)
input = [torch.tensor(x) for x in df["Patient"]]
output = [torch.tensor(y) for y in df["Doctor"]]

# %%
#Setup the training input and output
input = input[7300:]
output = output[7300:]

# %%
#Build the model
class pos_encoder(nn.Module):
  def __init__(self, embedding_size, max_len):
    super().__init__()
    self.pos_mat = torch.zeros(max_len, embedding_size)

    position = torch.arange(0, max_len, step = 1).float().unsqueeze(1)
    embedding_value = torch.arange(0, embedding_size, step = 2).float()
    div_term = 1/torch.tensor(100000.0)**(embedding_value/embedding_size)
    self.pos_mat[:, 0::2] = torch.sin(position*div_term)
    self.pos_mat[:, 1::2] = torch.cos(position*div_term)
    print(self.pos_mat)

  def forward(self, x):
    print(self.pos_mat[:x.size(0), :].shape)
    print(x.shape)
    return x + self.pos_mat[:x.size(0), :]


class attention(nn.Module):
  def __init__(self, embedding_size):
    super().__init__()
    self.w_query = nn.Linear(embedding_size, embedding_size, bias = False)
    self.w_key = nn.Linear(embedding_size, embedding_size, bias = False)
    self.w_value = nn.Linear(embedding_size, embedding_size, bias = False)

  def forward(self, x, mask):
    query = self.w_query(x)
    key = self.w_key(x)
    value = self.w_value(x)
    similarity = torch.matmul(query, key.transpose(0, 1))

    if mask != None:
      similarity = similarity.masked_fill(mask = mask, value = -1e9)

    attention_percentage = torch.nn.functional.softmax(similarity, dim = 0)
    attention_score = torch.matmul(attention_percentage, value)
    return attention_score

class residual_connection(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, position_value):
    return x + position_value
  

    
  
class Transformer(nn.Module):
  def __init__(self, input_vocab, embedding_size, max_len = 1000):
    super().__init__()
    self.embedding = nn.Embedding(input_vocab, embedding_size)
    self.pos_encoder = pos_encoder(embedding_size, max_len)
    self.attention = attention(embedding_size)
    self.residual_connection = residual_connection()
    self.embedding_size = embedding_size
    self.max_len = max_len
    self.fc = nn.Linear(embedding_size, input_vocab)

  def forward(self, x):
    out = self.embedding(x)
    pos = self.pos_encoder(out)
    mask = torch.tril(torch.ones(x.size(0), x.size(0)))
    mask = mask == 0
    out = self.attention(pos, mask)
    out = self.residual_connection(out, pos)
    out = self.fc(out)
    return out



# %%
model = Transformer(len(text_vocabulary), 100)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# %%
check_point = torch.load('Model_Checkpoint/check-point-after-adjusted-optimizer-and-restart-model.pt')
model.load_state_dict(check_point['model_state_dict'])
optimizer.load_state_dict(check_point['optimizer_state_dict'])

# %%
#Train the model


def train():
    num_epoch = 1
    for epoch in range(num_epoch):
        train_one_epoch()

def train_one_epoch():
    count_data = 0
    count_word = 0
    l_total = 0
    for i in range(len(input)):
        input_tokens = input[i]
        prediction = model(input_tokens)
        result = prediction[-1, :]
        result_id = torch.tensor([torch.argmax(result)])
        index = 0
        output_list = result_id
        for j in range(len(output[i])):
            if result_id.item() == text_vocabulary[eos_token]:
                break
            expect_output = torch.cat([input_tokens[1:], torch.tensor([output[i][0]])], dim = 0)
            print(prediction.shape)
            print(expect_output.shape)
            l = loss(prediction, expect_output)
            
            l_total += l
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            count_word += 1
            input_tokens = torch.concat([input_tokens, torch.tensor([output[i][j]])], dim = 0)
            prediction = model(input_tokens)
            result = prediction[-1, :]
            result_id = torch.tensor([torch.argmax(result)])
            output_list = torch.cat([output_list, result_id], dim = 0)
            index += 1
        count_data += 1
        print(f"Loss of word {count_data} is: {l_total/count_word} ")
        if count_data % 10 == 0:
            torch.save({
            'num_data': count_data,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': l_total/count_word,
            }, "check-point-after-adjusted-optimizer-and-restart-model.pt")
            
        
train()
    
    
    

# %%
input_test = "What is pregnancy"
input_test = get_rid_of_q(input_test)
input_test = clean_text(input_test)
input_test = add_eos(input_test)
input_test = token_to_index(input_test)
print(input_test)
input_test = torch.tensor(input_test)
prediction = model(input_test)
result = prediction[-1, :]
result_id = torch.tensor([torch.argmax(result)])
output_list = result_id
for i in range(len(input_test), 1000):
    if result_id.item() == text_vocabulary[eos_token]:
            break
    input_test = torch.concat([input_test, result_id], dim = 0)
    prediction = model(input_test)
    result = prediction[-1, :]
    result_id = torch.tensor([torch.argmax(result)])
    output_list = torch.cat([output_list, result_id], dim = 0)
    
print(type(output_list))
final = text_vocabulary.lookup_tokens(output_list.tolist())
print(' '.join(final))
    
    




