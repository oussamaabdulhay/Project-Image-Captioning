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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size)
        
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        
       
    
    def forward(self, features, captions):
        
        captions = captions[:, :-1] 
        
        captions_embeded = self.word_embeddings(captions)
        
             
        concatenate= torch.cat((features.unsqueeze(1), captions_embeded), dim=1)
        
        lstm_out,_ = self.lstm(concatenate)
        
        output = self.hidden2tag(lstm_out)
        
        return output
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        counter = 0
        final_output_list = [];
        print(inputs.shape)
        while (counter < max_len):
            lstm_out, _ = self.lstm(inputs)

            output = self.hidden2tag(lstm_out)
            temp = output.squeeze(0)
            temp = temp.squeeze(0)
            output_list = [ int(x) for x in temp]
            predicted_value = max(output_list)
            final_output_list.append(predicted_value)
            inputs = None
            inputs = [1,1, predicted_value]
            
            inputs = torch.FloatTensor(inputs)
            inputs = torch.tensor(inputs).to(torch.int64)
            inputs = self.word_embeddings(inputs)
            counter +=1
        print(final_output_list)
        return final_output_list