#1 Load dataset
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import datetime
import torch, pandas as pd, numpy as np
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy
"""
frequency-content 
"""

class freq_classifier(nn.Module):
    
    def __init__(self, input_size, hidden_dim, num_class,num_layer,device=None): 
        if device:
            self.device=device
        else:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(freq_classifier, self).__init__()

        self.rnn = nn.RNN(input_size,hidden_dim,num_layer,batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.hidden_to_popularity = nn.Linear(hidden_dim,num_class)
    def forward(self,feature,input_len):
        packed_input = pack_padded_sequence(feature,input_len,batch_first=True)
        packed_output, hidden_state = self.rnn(packed_input)
        #zero init start
        batch_output,_ = pad_packed_sequence(packed_output, batch_first=True)
        mask= (batch_output!=0).to(self.device).float()
        avg_pool_val = (batch_output*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
        #
        
        output = self.hidden_to_popularity(avg_pool_val)

        return output
"""
Visual-content 
"""
import datetime
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import glob,os
import pandas as pd
import numpy as np
import torchvision.models as models
import cv2
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN_LSTM_popularity_classifier(nn.Module):
    
    def __init__(self, input_size, hidden_dim, num_class,num_layer,device=None): 
        super(CNN_LSTM_popularity_classifier, self).__init__()
        if device:
            self.device=device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.resnet18(pretrained=True) # 224*224 
        
        self.lstm = nn.LSTM(input_size,hidden_dim,num_layer,batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.hidden_to_popularity = nn.Linear(hidden_dim,num_class)
    def forward(self,clips,input_len):
        batch_size, max_seq, c, w, h = clips.shape
        out = self.cnn(clips.view(batch_size*max_seq,c,w,h)).view(batch_size,max_seq,1000)
        out = pack_padded_sequence(out,input_len,batch_first=True)
        self.lstm.flatten_parameters()
        out, (_,_)= self.lstm(out)
        del _
        out,_ = pad_packed_sequence(out, batch_first=True)
        mask= (out!=0).to(self.device).float()
        out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
        del mask
        out = self.hidden_to_popularity(out)

        return out
"""
chat-content
"""
class LSTM_popularity_classifier(nn.Module):
    
    def __init__(self, input_size, hidden_dim, num_class,num_layer,device=None,wv_model=None,wv_model_path=None): 
        super(LSTM_popularity_classifier, self).__init__()
        if device:
            self.device=device
        else:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if wv_model is None:
            wv_model = Word2Vec.load(wv_mocel_path)
        print('max vocab size:{}'.format(wv_model.wv.vectors.shape[0]))
        self.embedding= nn.Embedding(wv_model.wv.vectors.shape[0], wv_model.wv.vectors.shape[1])
        self.embedding.weight= nn.Parameter(torch.Tensor(wv_model.wv.vectors))
        
        self.lstm = nn.LSTM(input_size,hidden_dim,num_layer,batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.hidden_to_popularity = nn.Linear(hidden_dim,num_class)
    def forward(self,vecs,input_len):
        vecs = self.embedding(vecs.to(torch.int64).to(self.device))
        packed_input = pack_padded_sequence(vecs.to(self.device),input_len,batch_first=True)
        packed_output, (hidden_state,cell_state)= self.lstm(packed_input)

        batch_output,_ = pad_packed_sequence(packed_output, batch_first=True)
        mask= (batch_output!=0).to(self.device).float()
        avg_pool_val = (batch_output*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values

        output = self.hidden_to_popularity(avg_pool_val)
        
        return output
"""
ensemble model
"""
class Ensemble_model(nn.Module):
    def __init__(self,settings= {}):
        super(Ensemble_model, self).__init__()
        """
        settings = {'freq':
                    {'set':bool,
                     'input_size',int, ...}
                    'chat':{'set':bool, ...}}
        """
        self.settings = settings
        self.flags = [settings.get('chat',{}).get('set'),settings.get('visual',{}).get('set'),settings.get('freq',{}).get('set')]
        num_out = 0
        if self.settings.get('chat',{'set':False}).get('set',False):
            c_set = self.settings['chat']
            input_size, hidden_dim, num_class,num_layer = c_set['input_size'],c_set['hidden_dim'],c_set['num_class'],c_set['num_layer']
            # 'embeddings/sg_emoteonly_textonly_intersect_w2v.model'
            wv_model = Word2Vec.load(c_set['embedding_path'])
            self.embedding_chat=  nn.Embedding(wv_model.wv.vectors.shape[0], wv_model.wv.vectors.shape[1])
            self.embedding_chat.weight= nn.Parameter(torch.Tensor(wv_model.wv.vectors))
            
            self.lstm_chat = nn.LSTM(input_size,hidden_dim,num_layer,batch_first=True)
            self.w1 = nn.Linear(128,1)
            self.w4 = nn.Linear(128,1)
            num_out +=1
        if self.settings.get('visual',{'set':False}).get('set',False):
            c_set = self.settings['visual']
            self.cnn_visual = models.resnet18(pretrained=True) # 224*224 
            input_size, hidden_dim, num_class,num_layer = c_set['input_size'],c_set['hidden_dim'],c_set['num_class'],c_set['num_layer']
            self.lstm_visual= nn.LSTM(input_size,hidden_dim,num_layer,batch_first=True)
            self.w2 = nn.Linear(128,1)
            self.w5 = nn.Linear(128,1)
            num_out +=1
        if self.settings.get('freq',{'set':False}).get('set',False):
            c_set = self.settings['freq']
            input_size, hidden_dim, num_class,num_layer = c_set['input_size'],c_set['hidden_dim'],c_set['num_class'],c_set['num_layer']
            self.rnn_freq = nn.LSTM(input_size,hidden_dim,num_layer,batch_first=True)
            self.w3 = nn.Linear(128,1)
            self.w6 = nn.Linear(128,1)
            num_out+=1
        self.sm = nn.Softmax(dim=1)
        self.fc = nn.Linear(num_out,2)
    def load_pretrained_model(self,chat,video,freq):
        if chat:
            self.embedding_chat = copy.deepcopy(chat.embedding)
            self.lstm_chat = copy.deepcopy(chat.lstm)
        if video:
            self.cnn_visual = copy.deepcopy(video.cnn)
            self.lstm_visual = copy.deepcopy(video.lstm)
        if freq:
            self.rnn_freq = copy.deepcopy(freq.rnn)
    def forward_history(self,chat_input,visual_input,freq_input):
        outs = []
        outs2 = []
        if self.flags[0]: # chat
            input_chat,input_len = chat_input
            out = self.embedding_chat(input_chat.to(torch.int64))
            out = pack_padded_sequence(out.to(self.w1.weight.device),input_len,batch_first=True)
            out, (hidden_state,cell_state)= self.lstm_chat(out)
            out,_ = pad_packed_sequence(out, batch_first=True)
            mask= (out!=0).to(self.w1.weight.device).float()
            out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
            out2 = self.w4(out)
            out = self.w1(out)
            
            outs.append(out)
            outs2.append(out2)
        if self.flags[1]: #visual
            clips,input_len = visual_input
            batch_size, max_seq, c, w, h = clips.shape
            out = self.cnn_visual(clips.view(batch_size*max_seq,c,w,h)).view(batch_size,max_seq,1000)
            out = pack_padded_sequence(out,input_len,batch_first=True)
            self.lstm_visual.flatten_parameters()
            out, (_,_)= self.lstm_visual(out)
            out,_ = pad_packed_sequence(out, batch_first=True)
            mask= (out!=0).to(self.w2.weight.device).float()
            out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
            out2 = self.w5(out)
            out = self.w2(out)
            
            outs.append(out)
            outs2.append(out2)
        if self.flags[2]: # freq
            feature,input_len = freq_input
            out = pack_padded_sequence(feature,input_len,batch_first=True)
            out, hidden_state = self.rnn_freq(out)
            #zero init start
            out,_ = pad_packed_sequence(out, batch_first=True)
            mask= (out!=0).to(self.w3.weight.device).float()
            out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
            out2 = self.w6(out)
            out = self.w3(out)
            outs.append(out)
            outs2.append(out2)
        
        out = torch.cat(outs,dim=1)
        
        out2 = torch.cat(outs2,dim=1)
        out2 = self.sm(out2)
        final_out = out*out2
        final_out = final_out.reshape([1,final_out.shape[-1]])
           
        final_out = self.fc(final_out)
        return out,out2,final_out 
    def forward(self,chat_input,visual_input,freq_input):

        outs = []
        outs2 = []
        if self.flags[0]: # chat
            input_chat,input_len = chat_input
            out = self.embedding_chat(input_chat.to(torch.int64))
            out = pack_padded_sequence(out.to(self.w1.weight.device),input_len,batch_first=True)
            out, (hidden_state,cell_state)= self.lstm_chat(out)
            out,_ = pad_packed_sequence(out, batch_first=True)
            mask= (out!=0).to(self.fc.weight.device).float()
            out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
            out2 = self.w4(out)
            out = self.w1(out)
            
            outs.append(out)
            outs2.append(out2)
        if self.flags[1]: #visual
            clips,input_len = visual_input
            batch_size, max_seq, c, w, h = clips.shape
            out = self.cnn_visual(clips.view(batch_size*max_seq,c,w,h)).view(batch_size,max_seq,1000)
            out = pack_padded_sequence(out,input_len,batch_first=True)
            self.lstm_visual.flatten_parameters()
            out, (_,_)= self.lstm_visual(out)
            out,_ = pad_packed_sequence(out, batch_first=True)
            mask= (out!=0).to(self.w2.weight.device).float()
            out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
            out2 = self.w5(out)
            out = self.w2(out)
            
            outs.append(out)
            outs2.append(out2)
        if self.flags[2]: # freq
            feature,input_len = freq_input
            out = pack_padded_sequence(feature,input_len,batch_first=True)
            out, hidden_state = self.rnn_freq(out)
            #zero init start
            out,_ = pad_packed_sequence(out, batch_first=True)
            mask= (out!=0).to(self.w3.weight.device).float()
            out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
            out2 = self.w6(out)
            out = self.w3(out)
            outs.append(out)
            outs2.append(out2)
        
        out = torch.cat(outs,dim=1)
        
        out2 = torch.cat(outs2,dim=1)
        out2 = self.sm(out2)
        out = out*out2
        out = out.reshape([1,out.shape[-1]])
        
        out = self.fc(out)
        return out
    
"""
Feature-based model Image
"""
class FB_img(nn.Module):
    def __init__(self,device=None):
        super(FB_img, self).__init__()
        input_dim = 1000
        output_dim = 2
        self.vgg = models.vgg16(pretrained=True)
        self.fc = torch.nn.Linear(input_dim,output_dim)

    def forward(self,clips,input_len):
        batch_size, max_seq, c, w, h = clips.shape
        with torch.no_grad():
            out = self.vgg(clips.view(batch_size*max_seq,c,w,h))
            out = out.view(batch_size,max_seq,1000)
            out = pack_padded_sequence(out.to(self.fc.weight.device),input_len,batch_first=True)
            out ,_ = pad_packed_sequence(out, batch_first=True)
            mask= (out!=0).to(self.fc.weight.device).float()
            out = (out*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
        out = self.fc(out)
        return out
    def backward(self,out,target):
        self.optimizer.zero_grad()
        loss = self.criterion(out,target)
        loss.backward()
        self.optimizer.step()
        return loss
"""
Featurebased model freq
"""
class FB_freq(nn.Module):
    def __init__(self,device=None): 

        super(FB_freq, self).__init__()

        if device:
            self.device=device
        else:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_to_popularity = nn.Linear(5,2)

    def forward(self,vecs):
        output = self.hidden_to_popularity(vecs)
        return output
"""
Feature-based model chat
"""
class FB_text(nn.Module):
    def __init__(self,device=None,wv_model=None,wv_model_path=None): 
        #embedding dim should be 100, num_class should be 10
        super(FB_text, self).__init__()
        # embedding initialization with pretrained vectors. 
        if device:
            self.device=device
        else:
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if wv_model is None:
            wv_model = Word2Vec.load(wv_model_path)
        self.embedding= nn.Embedding(wv_model.wv.vectors.shape[0], wv_model.wv.vectors.shape[1])
        self.embedding.weight= nn.Parameter(torch.Tensor(wv_model.wv.vectors))
        self.embedding.weight.requires_grad = False
        self.hidden_to_popularity = nn.Linear(100,2)
    def forward(self,vecs,input_len):
        vecs = self.embedding(vecs.to(torch.int64))
        packed_input = pack_padded_sequence(vecs.to(self.device),input_len,batch_first=True)
        batch_output,_ = pad_packed_sequence(packed_input, batch_first=True)
        mask= (batch_output!=0).to(self.device).float()
        avg_pool_val = (batch_output*mask).sum(dim=1)/mask.sum(dim=1) # mean value without zero values
        output = self.hidden_to_popularity(avg_pool_val.to(device))
        return output