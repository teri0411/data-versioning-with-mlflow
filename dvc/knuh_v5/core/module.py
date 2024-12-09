import pickle

import torch
from torch import nn
import torch.nn.functional as F

RECURRENT = nn.GRU

# if torch.cuda.is_available: DEVICE = torch.device("cuda")
# else: DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu")

class Encoder(nn.Module):
    def __init__(self,type,n,m,l,do): # 
        super(Encoder,self).__init__()
        print(f"type임: {type}")
        self.type = type # recurrent neural network type
        self.n = n # input dimension
        self.m = m # hidden dimension
        self.l = l # 
        self.drop_out = do
        
        self.lstm = RECURRENT(self.n, self.m,self.l,dropout=self.drop_out,batch_first=True)

    def forward(self,X): # (batch_size,T,n)
        batch_size,T,m = X.shape
        
        hidden = self.init_hidden(batch_size)
        
        output,hidden = self.lstm(X,hidden)

        return output,hidden

    def init_hidden(self,batch_size=1):
        if self.type == "lstm":
            return (torch.zeros(self.l,batch_size,self.m,device=DEVICE),
                torch.zeros(self.l,batch_size,self.m,device=DEVICE))
        else:
            return torch.zeros(self.l,batch_size,self.m,device=DEVICE)
        
class Decoder(nn.Module):
    def __init__(self,type,o,m,l,l_m,do,fc_do):
        super(Decoder,self).__init__()
        
        self.type = type
        self.o = o   #output dimension
        self.m = m
        self.l = l
        self.l_m = l_m
        self.drop_out = do
        self.fc_drop_out = fc_do
        
        self.lstm = RECURRENT(self.o,self.m,self.l,dropout=self.drop_out,batch_first=True)
        
        #Output layer
        self.fc1 = nn.Linear(self.m,self.l_m)
        self.layernorm = nn.LayerNorm(self.l_m)
        self.dropout = nn.Dropout(self.drop_out)
        
        self.fc2 = nn.Linear(self.l_m,self.o)

    def forward(self,inputs, hidden,encoder_outputs):
        output, hidden = self.lstm(inputs.float(),hidden)
        output = output.squeeze(1)
        
        output = self.fc1(output)
        output = self.layernorm(output)
        output = F.gelu(output)
        output = self.dropout(output)
        
        output = self.fc2(output)

        return output, hidden, None
    
class Seq2Seq(nn.Module):
    def __init__(self,type,model_params,T,test=False):
        super(Seq2Seq,self).__init__()
        
        self.T = T
        self.type = type
        self.encoder = Encoder(**model_params['encoder'])
        self.test = test
        
        self.decoder = Decoder(**model_params['decoder'])
        
    def forward(self,X,Y0):
        batch_size,t,m = X.shape
        
        outputs = []    
        X_encoded,hidden =  self.encoder(X)
        
        for t in range(self.T):
            output,hidden,none = self.decoder(Y0,hidden,X_encoded)
            
            outputs.append(output)
            Y0 = torch.softmax(output.unsqueeze(1),-1)
        
        outputs = torch.stack(outputs,dim=1).squeeze(1)
        if self.test: outputs = torch.softmax(outputs,-1)
        
        return outputs
    
def load_pickle(path):
    with open(path,'rb') as f:
        load_data = pickle.load(f)
    return load_data