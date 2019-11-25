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

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #inititializing lstm
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        ## TODO: define the final, fully-connected output layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
     # initialize the weights
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.word_embeddings.weight)
    
    
    def forward(self, features, captions):
        embeds = self.word_embeddings(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)
        
        x, (h, c) = self.lstm(inputs)
        
        # Stack up LSTM outputs using view
        #x = x.view(x.size()[0]*x.size()[1], self.n_hidden)
        
        ## TODO: put x through the fully-connected layer
        x = self.linear(x)
        
        # return x and the hidden state (h, c)
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        preds = []
        count = 0
        word_item = None
        
        while count < max_len and word_item != 1 :
            
            #Predict output
            output_lstm, states = self.lstm(inputs, states)
            output = self.linear(output_lstm.squeeze(1))
            
            #Get max value
            prob, word = output.max(1)
            
            #append word
            word_item = word.item()
            preds.append(word_item)
            
            #next input is current prediction
            inputs = self.word_embeddings(word).unsqueeze(1)
            
            count+=1
        
        return preds