import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        #self.bn = nn.BatchNorm1d(embed_size)  #Added BatchNormalization

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        #features = self.bn(features) #Added BatchNormalization
        return features
    

# Some code snippets from lesson 4 LSTM: LSTM for Part of Speech Tagging
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)#(batch, seq, feature)
        
        # the linear layer that maps the hidden state output dimension to the vocab size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        
    def init_hidden(batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
    
    def forward(self, features, captions):
        captions = captions [:, :-1] #(batch , caption_len - 1)
        embeddings = self.embed(captions) #(batch, caption_len -1, embed_size)
        features = features.unsqueeze(1) # (batch, 1, embed_size)
        inputs =  torch.cat((features, embeddings), dim=1) # (batch, caption_len , embed_size)
        lstm_output, (hn, cn) = self.lstm(inputs) #(batch, caption_len, 1* hidden_size)
        output = self.fc(lstm_output) # (batch,caption_len,vocab_size)
        return output

        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ids = []
        for i in range(max_len):
            out, hidden = self.lstm(inputs, states) # (1, 1, hidden_size) ->((1, 1, hidden_size), (1, 1, hidden_size))
            out = self.fc(out.squeeze(1)) #(1, vocab_size) -> ((1, hidden_size))
            max_indice = out.max(1)[1]
            new_id = max_indice.to("cpu").item()
            ids.append(new_id)
            if new_id == 1:
                break
            inputs = self.embed(new_id) # (1, embed_size)
            inputs = inputs.unsqueeze(1) # (1, 1, embed_size)
            
        return ids