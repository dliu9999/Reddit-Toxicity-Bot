import spacy
from torchtext import data
import torch
import torch.nn as nn

# declare model class
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        embedded = self.dropout(self.embedding(x))
        
        output, (hidden, cell) = self.rnn(embedded)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
            
        return self.fc(hidden.squeeze(0))


# load vocabulary
PATH1 = "../RNN_1/vocab_win.pt"
comment = torch.load(PATH1)

PATH2 = "../RNN_1/model.pt"
INPUT_DIM = len(comment.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

# create model object
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

# load saved model and evaluate
model.load_state_dict(torch.load(PATH2, map_location='cpu'))
model.eval()

NLP = spacy.load('en')
def predict_sentiment(sentence):
    tokenized = [tok.text for tok in NLP.tokenizer(sentence)] # tokenize input sentence
    indexed = [comment.vocab.stoi[t] for t in tokenized]      # index the tokens
    tensor = torch.LongTensor(indexed)                        # convert tokens to PyTorch tensors
    tensor = tensor.unsqueeze(1)                              # add batch dimension
    prediction = torch.sigmoid(model(tensor))                 # apply on linearity
    return prediction.item()                                  # converts tensor to af float