import torch
import torch.nn as nn
from torchcrf import CRF
from data_loader import WebsiteProcessor
from Transformer import TransformerBlock,CoTransformerBlock
from transformers import BertModel, BertTokenizer
import numpy as np

class CharCNN(nn.Module):
    def __init__(self,
                 max_word_len=30,
                 kernel_lst="2,3,4",
                 num_filters=32,
                 char_vocab_size=1000,
                 char_emb_dim=30,
                 final_char_dim=50):
        super(CharCNN, self).__init__()

        # Initialize character embedding
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -0.25, 0.25)

        kernel_lst = list(map(int, kernel_lst.split(",")))  # "2,3,4" -> [2, 3, 4]

        # Convolution for each kernel
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size // 2),
                nn.Tanh(),  # As the paper mentioned
                nn.MaxPool1d(max_word_len - kernel_size + 1),
                nn.Dropout(0.25)  # As same as the original code implementation
            ) for kernel_size in kernel_lst
        ])

        self.linear = nn.Sequential(
            nn.Linear(num_filters * len(kernel_lst), 100),
            nn.ReLU(),  # As same as the original code implementation
            nn.Dropout(0.25),
            nn.Linear(100, final_char_dim)
        )

    def forward(self, x):
        """
        :param x: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, final_char_dim)
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_emb(x)  # (b, s, w, d)
        x = x.view(batch_size * max_seq_len, max_word_len, -1)  # (b*s, w, d)
        x = x.transpose(2, 1)  # (b*s, d, w): Conv1d takes in (batch, dim, seq_len), but raw embedded is (batch, seq_len, dim)

        conv_lst = [conv(x) for conv in self.convs]
        conv_concat = torch.cat(conv_lst, dim=-1)  # (b*s, num_filters, len(kernel_lst))
        conv_concat = conv_concat.view(conv_concat.size(0), -1)  # (b*s, num_filters * len(kernel_lst))

        output = self.linear(conv_concat)  # (b*s, final_char_dim)
        output = output.view(batch_size, max_seq_len, -1)  # (b, s, final_char_dim)
        return output


class Bert_BiLSTM(nn.Module):
    def __init__(self, args):
        super(Bert_BiLSTM, self).__init__()
        self.args = args
        # 这里我们调用bert-base模型，同时模型的词典经过小写处理
        model_name = 'bert-base-uncased'
        # 读取模型对应的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 载入模型
        self.bert =  BertModel.from_pretrained(model_name).to('cuda:0')

        #print(self.bert.device)
        self.bi_lstm = nn.LSTM(input_size=768,
                               hidden_size=args.hidden_dim // 2,  # Bidirectional will double the hidden_size
                               bidirectional=True,
                               batch_first=True)

    def bert_embeddings(self,token_ids,token_length,max_seq_len):
        #print(token_ids,token_length)
        total_embs = []
        for input_ids,token_len in zip(token_ids,token_length):
            # 通过tokenizer把文本变成 token_id
            #print(input_ids,token_len)
            input_ids = list(input_ids[:token_len].cpu().numpy())
            #print(input_ids)
            #print(len(input_ids))
            # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)[1:-1]
            #print(token_len)
            word2token_dict={}
            word_idx = 0
            word2token_dict[word_idx]=[0]
            for i in range(1,len(input_tokens)):
                input_token = input_tokens[i]
                if  '##' in input_token:
                    word2token_dict[word_idx].append(i)
                else:
                    word_idx = word_idx + 1
                    word2token_dict[word_idx] = [i]
            #if len(word2token_dict.keys())>5 and len(word2token_dict.keys())<35:
            #    print(word2token_dict)
            #    print(input_tokens)
            input_ids = torch.tensor([input_ids]).to('cuda:0')
            #print(input_ids.device)
            with torch.no_grad():
                last_hidden_states = self.bert(input_ids)[0]  # Models outputs are now tuples
                last_hidden_states = torch.squeeze(last_hidden_states,dim=0)
                last_hidden_states = last_hidden_states[1:-1,:]
                
            embs = []
            for word_idx in word2token_dict.keys():
                token_ids = word2token_dict[word_idx]
                emb = []
                for token_id in token_ids:
                    emb.append(last_hidden_states[token_id,:].cpu().detach().numpy())
                #emb = torch.tensor(emb)
                emb = np.mean(emb,axis=0)
                embs.append(emb)
            total_embs.append(embs[:max_seq_len])            
        return torch.tensor(total_embs).to('cuda:0')

    def forward(self, token_ids,token_length,max_seq_len):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, dim)
        """
        bert_embs = self.bert_embeddings(token_ids,token_length,max_seq_len)
        #print(bert_embs.size())
        #print(bert_embs.device)
        lstm_output, _ = self.bi_lstm(bert_embs, None)
        return lstm_output


class BiLSTM(nn.Module):
    def __init__(self, args, pretrained_word_matrix):
        super(BiLSTM, self).__init__()
        self.args = args
        self.char_cnn = CharCNN(max_word_len=args.max_word_len,
                                kernel_lst=args.kernel_lst,
                                num_filters=args.num_filters,
                                char_vocab_size=args.char_vocab_size,
                                char_emb_dim=args.char_emb_dim,
                                final_char_dim=args.final_char_dim)
        if pretrained_word_matrix is not None:
            self.word_emb = nn.Embedding.from_pretrained(pretrained_word_matrix)
        else:
            self.word_emb = nn.Embedding(args.word_vocab_size, args.word_emb_dim, padding_idx=0)
            nn.init.uniform_(self.word_emb.weight, -0.25, 0.25)

        self.bi_lstm = nn.LSTM(input_size=args.word_emb_dim + args.final_char_dim,
                               hidden_size=args.hidden_dim // 2,  # Bidirectional will double the hidden_size
                               bidirectional=True,
                               batch_first=True)

    def forward(self, word_ids, char_ids):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, dim)
        """
        w_emb = self.word_emb(word_ids)
        c_emb = self.char_cnn(char_ids)

        w_c_emb = torch.cat([w_emb, c_emb], dim=-1)

        lstm_output, _ = self.bi_lstm(w_c_emb, None)
        return lstm_output


class CoAttention_DNSImage(nn.Module):
    def __init__(self, args):
        super(CoAttention_DNSImage, self).__init__()
        self.args = args

        # linear for word-guided visual attention
        self.dns_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.img_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.att_linear_1 = nn.Linear(args.hidden_dim * 2, 1)

        # linear for visual-guided textual attention
        self.dns_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.img_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, dns_feature, img_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param img_features: (batch_size, num_img_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_img_features (batch_size, max_seq_len, hidden_dim)
        """
        ############### 1. Word-guided visual attention ###############
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        img_features_rep = img_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len_dns, 1)
        dns_features_rep = dns_feature.unsqueeze(1).repeat(1, self.args.num_img_region, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        dns_features_rep = self.dns_linear_1(dns_features_rep)
        img_features_rep = self.img_linear_1(img_features_rep)

        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_img_region, hidden_dim * 2]
        concat_features = torch.cat([img_features_rep,dns_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_img_region]
        textual_att = self.att_linear_1(concat_features).squeeze(-1)
        textual_att = torch.softmax(textual_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_dns_features = torch.matmul(textual_att, dns_feature)  # Vt_hat

        ############### 2. Visual-guided textual Attention ###############
        # 2.1 Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        dns_features_rep = att_dns_features.unsqueeze(2).repeat(1, 1, self.args.num_img_region, 1)
        img_features_rep = img_features.unsqueeze(1).repeat(1, self.args.num_img_region, 1, 1)

        # 2.2 Feed to single layer (d*k) -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = self.img_linear_2(img_features_rep)
        dns_features_rep = self.dns_linear_2(dns_features_rep)

        # 2.3. Concat & tanh -> [batch_size, max_seq_len, max_seq_len, hidden_dim * 2]
        concat_features = torch.cat([dns_features_rep,img_features_rep ], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, max_seq_len]
        visual_att = self.att_linear_2(concat_features).squeeze(-1)
        visual_att = torch.softmax(visual_att, dim=-1)

        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        att_img_features = torch.matmul(visual_att, img_features)  # Ht_hat

        return att_img_features,att_dns_features



class CoAttention_ImageDNS(nn.Module):
    def __init__(self, args):
        super(CoAttention_ImageDNS, self).__init__()
        self.args = args

        # linear for word-guided visual attention
        self.dns_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.img_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.att_linear_1 = nn.Linear(args.hidden_dim * 2, 1)

        # linear for visual-guided textual attention
        self.dns_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.img_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, dns_feature, img_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param img_features: (batch_size, num_img_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_img_features (batch_size, max_seq_len, hidden_dim)
        """
        ############### 1. Word-guided visual attention ###############
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        dns_features_rep = dns_feature.unsqueeze(2).repeat(1, 1, self.args.num_img_region, 1)
        img_features_rep = img_features.unsqueeze(1).repeat(1, self.args.max_seq_len_dns, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        dns_features_rep = self.dns_linear_1(dns_features_rep)
        img_features_rep = self.img_linear_1(img_features_rep)

        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_img_region, hidden_dim * 2]
        concat_features = torch.cat([dns_features_rep, img_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_img_region]
        visual_att = self.att_linear_1(concat_features).squeeze(-1)
        visual_att = torch.softmax(visual_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_img_features = torch.matmul(visual_att, img_features)  # Vt_hat

        ############### 2. Visual-guided textual Attention ###############
        # 2.1 Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = att_img_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len_dns, 1)
        dns_features_rep = dns_feature.unsqueeze(1).repeat(1, self.args.max_seq_len_dns, 1, 1)

        # 2.2 Feed to single layer (d*k) -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = self.img_linear_2(img_features_rep)
        dns_features_rep = self.dns_linear_2(dns_features_rep)

        # 2.3. Concat & tanh -> [batch_size, max_seq_len, max_seq_len, hidden_dim * 2]
        concat_features = torch.cat([img_features_rep, dns_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, max_seq_len]
        textual_att = self.att_linear_2(concat_features).squeeze(-1)
        textual_att = torch.softmax(textual_att, dim=-1)

        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        att_dns_features = torch.matmul(textual_att, dns_feature)  # Ht_hat

        return att_dns_features, att_img_features


class CoAttention_TextImage(nn.Module):
    def __init__(self, args):
        super(CoAttention_TextImage, self).__init__()
        self.args = args

        # linear for word-guided visual attention
        self.text_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.img_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.att_linear_1 = nn.Linear(args.hidden_dim * 2, 1)

        # linear for visual-guided textual attention
        self.text_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.img_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, text_features, img_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param img_features: (batch_size, num_img_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_img_features (batch_size, max_seq_len, hidden_dim)
        """
        ############### 1. Word-guided visual attention ###############
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = text_features.unsqueeze(2).repeat(1, 1, self.args.num_img_region, 1)
        img_features_rep = img_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = self.text_linear_1(text_features_rep)
        img_features_rep = self.img_linear_1(img_features_rep)

        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_img_region, hidden_dim * 2]
        concat_features = torch.cat([text_features_rep, img_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_img_region]
        visual_att = self.att_linear_1(concat_features).squeeze(-1)
        visual_att = torch.softmax(visual_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_img_features = torch.matmul(visual_att, img_features)  # Vt_hat

        ############### 2. Visual-guided textual Attention ###############
        # 2.1 Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = att_img_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len, 1)
        text_features_rep = text_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 2.2 Feed to single layer (d*k) -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        img_features_rep = self.img_linear_2(img_features_rep)
        text_features_rep = self.text_linear_2(text_features_rep)

        # 2.3. Concat & tanh -> [batch_size, max_seq_len, max_seq_len, hidden_dim * 2]
        concat_features = torch.cat([img_features_rep, text_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, max_seq_len]
        textual_att = self.att_linear_2(concat_features).squeeze(-1)
        textual_att = torch.softmax(textual_att, dim=-1)

        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        att_text_features = torch.matmul(textual_att, text_features)  # Ht_hat

        return att_text_features, att_img_features

class CoAttention_TextDNS(nn.Module):
    def __init__(self, args):
        super(CoAttention_TextDNS, self).__init__()
        self.args = args

        # linear for word-guided visual attention
        self.text_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.dns_linear_1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.att_linear_1 = nn.Linear(args.hidden_dim * 2, 1)

        # linear for visual-guided textual attention
        self.text_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.dns_linear_2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, text_features, dns_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param img_features: (batch_size, num_img_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_img_features (batch_size, max_seq_len, hidden_dim)
        """
        ############### 1. Word-guided visual attention ###############
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = text_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len_dns, 1)
        dns_features_rep = dns_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_img_region, hidden_dim]
        text_features_rep = self.text_linear_1(text_features_rep)
        dns_features_rep = self.dns_linear_1(dns_features_rep)

        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_img_region, hidden_dim * 2]
        concat_features = torch.cat([text_features_rep, dns_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_img_region]
        visual_att = self.att_linear_1(concat_features).squeeze(-1)
        visual_att = torch.softmax(visual_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_dns_features = torch.matmul(visual_att, dns_features)  # Vt_hat

        ############### 2. Visual-guided textual Attention ###############
        # 2.1 Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        dns_features_rep = att_dns_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len, 1)
        text_features_rep = text_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 2.2 Feed to single layer (d*k) -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        dns_features_rep = self.dns_linear_2(dns_features_rep)
        text_features_rep = self.text_linear_2(text_features_rep)

        # 2.3. Concat & tanh -> [batch_size, max_seq_len, max_seq_len, hidden_dim * 2]
        concat_features = torch.cat([dns_features_rep, text_features_rep], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, max_seq_len]
        textual_att = self.att_linear_2(concat_features).squeeze(-1)
        textual_att = torch.softmax(textual_att, dim=-1)

        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        att_text_features = torch.matmul(textual_att, text_features)  # Ht_hat

        return att_text_features, att_dns_features

class GMF(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim, args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, att_text_features, att_img_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]
        
        gate_img = self.gate_linear(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img = torch.sigmoid(gate_img)  # [batch_size, max_seq_len, 1]
        gate_img = gate_img.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img, new_img_feat) + torch.mul(1 - gate_img, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features
class GMF1(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF1, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim*2, args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(args.hidden_dim*2, args.hidden_dim)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, att_text_features, att_img_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]

    
        
        gate_img = self.gate_linear(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img = torch.sigmoid(gate_img)  # [batch_size, max_seq_len, 1]
        gate_img = gate_img.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img, new_img_feat) + torch.mul(1 - gate_img, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features
class GMF2(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF2, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim*2, args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.dns_linear = nn.Linear(args.hidden_dim*2, args.hidden_dim)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, att_text_features, att_dns_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_dns_feat = torch.tanh(self.dns_linear(att_dns_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]

    
        
        gate_dns = self.gate_linear(torch.cat([new_dns_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_dns = torch.sigmoid(gate_dns)  # [batch_size, max_seq_len, 1]
        gate_dns = gate_dns.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_dns, new_dns_feat) + torch.mul(1 - gate_dns, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features

class GMF3(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF3, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim*4 , args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(args.hidden_dim*4, args.hidden_dim)
        self.dns_linear = nn.Linear(args.hidden_dim*4, args.hidden_dim)
        self.gate_linear1 = nn.Linear(args.hidden_dim * 2, 1)
        self.gate_linear2 = nn.Linear(args.hidden_dim * 2, 1)
        self.gate_linear3 = nn.Linear(args.hidden_dim * 2, 1)
    def forward(self, att_text_features, att_img_features,att_dns_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]
        new_dns_feat = torch.tanh(self.dns_linear(att_dns_features))  # [batch_size, max_seq_len, hidden_dim]


        gate_img1 = self.gate_linear1(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img2 = self.gate_linear2(torch.cat([new_dns_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        


        gate_img1 = torch.sigmoid(gate_img1)  # [batch_size, max_seq_len, 1]
        gate_img2 = torch.sigmoid(gate_img2)  # [batch_size, max_seq_len, 1]

        gate_img1 = gate_img1.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        gate_img2 = gate_img2.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]


        multimodal_features1 = torch.mul(gate_img1, new_img_feat) + torch.mul(1 - gate_img1, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features2 = torch.mul(gate_img2, new_dns_feat) + torch.mul(1 - gate_img2, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        gate_img3 = self.gate_linear2(torch.cat([multimodal_features1, multimodal_features2], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img3 = torch.sigmoid(gate_img3)  # [batch_size, max_seq_len, 1]
        gate_img3 = gate_img3.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img3, multimodal_features1) + torch.mul(1 - gate_img3, multimodal_features2)  # [batch_size, max_seq_len, hidden_dim]
      
        return multimodal_features
class GMF4(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF4, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim*6 , args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(args.hidden_dim*4, args.hidden_dim)
        self.dns_linear = nn.Linear(args.hidden_dim*3, args.hidden_dim)
        self.gate_linear1 = nn.Linear(args.hidden_dim * 2, 1)
        self.gate_linear2 = nn.Linear(args.hidden_dim * 2, 1)
        self.gate_linear3 = nn.Linear(args.hidden_dim * 2, 1)
    def forward(self, att_text_features, att_img_features,att_dns_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]
        new_dns_feat = torch.tanh(self.dns_linear(att_dns_features))  # [batch_size, max_seq_len, hidden_dim]


        gate_img1 = self.gate_linear1(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img2 = self.gate_linear2(torch.cat([new_dns_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        


        gate_img1 = torch.sigmoid(gate_img1)  # [batch_size, max_seq_len, 1]
        gate_img2 = torch.sigmoid(gate_img2)  # [batch_size, max_seq_len, 1]

        gate_img1 = gate_img1.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        gate_img2 = gate_img2.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]


        multimodal_features1 = torch.mul(gate_img1, new_img_feat) + torch.mul(1 - gate_img1, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features2 = torch.mul(gate_img2, new_dns_feat) + torch.mul(1 - gate_img2, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        gate_img3 = self.gate_linear2(torch.cat([multimodal_features1, multimodal_features2], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img3 = torch.sigmoid(gate_img3)  # [batch_size, max_seq_len, 1]
        gate_img3 = gate_img3.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img3, multimodal_features1) + torch.mul(1 - gate_img3, multimodal_features2)  # [batch_size, max_seq_len, hidden_dim]
      
        return multimodal_features

class FiltrationGate(nn.Module):
    """
    In this part, code is implemented in other way compare to equation on paper.
    So I mixed the method between paper and code (e.g. Add `nn.Linear` after the concatenated matrix)
    """

    def __init__(self, args):
        super(FiltrationGate, self).__init__()
        self.args = args

        self.text_linear = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.multimodal_linear = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

        self.resv_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.output_linear = nn.Linear(args.hidden_dim * 2, len(WebsiteProcessor.get_labels()))

    def forward(self, text_features, multimodal_features):
        """
        :param text_features: Original text feature from BiLSTM [batch_size, max_seq_len, hidden_dim]
        :param multimodal_features: Feature from GMF [batch_size, max_seq_len, hidden_dim]
        :return: output: Will be the input for CRF decoder [batch_size, max_seq_len, hidden_dim]
        """
        # [batch_size, max_seq_len, 2 * hidden_dim]
        concat_feat = torch.cat([self.text_linear(text_features), self.multimodal_linear(multimodal_features)], dim=-1)
        # This part is not written on equation, but if is needed
        filtration_gate = torch.sigmoid(self.gate_linear(concat_feat))  # [batch_size, max_seq_len, 1]
        filtration_gate = filtration_gate.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]

        reserved_multimodal_feat = torch.mul(filtration_gate,
                                             torch.tanh(self.resv_linear(multimodal_features)))  # [batch_size, max_seq_len, hidden_dim]
        output = self.output_linear(torch.cat([text_features, reserved_multimodal_feat], dim=-1))  # [batch_size, max_seq_len, num_tags]

        return output


class ACN(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN, self).__init__()
        self.lstm = BiLSTM(args, pretrained_word_matrix)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.co_attention = CoAttention_TextImage(args)
        self.gmf = GMF(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids, img_feature, mask, label_ids):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """
        text_features = self.lstm(word_ids, char_ids)
        img_features = self.dim_match(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
        assert text_features.size(-1) == img_features.size(-1)

        att_text_features, att_img_features = self.co_attention(text_features, img_features)
        multimodal_features = self.gmf(att_text_features, att_img_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits


class ACN6(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    Bert -> LSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args):
        super(ACN6, self).__init__()
        
        self.lstm = Bert_BiLSTM(args)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.co_attention = CoAttention_TextImage(args)
        self.gmf = GMF(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)
        self.args = args

    def forward(self, img_feature, mask, label_ids,token_ids,token_length):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """
        text_features = self.lstm(token_ids,token_length,self.args.max_seq_len)
        img_features = self.dim_match(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
        assert text_features.size(-1) == img_features.size(-1)

        att_text_features, att_img_features = self.co_attention(text_features, img_features)
        multimodal_features = self.gmf(att_text_features, att_img_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits

class ACN7(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN7, self).__init__()
        self.args = args
        self.lstm1 = Bert_BiLSTM(args)
        self.lstm2 = Bert_BiLSTM(args)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        
        device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        is_cuda = True if torch.cuda.is_available() and not args.no_cuda else False

        self.text_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.img_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.dns_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
       
        self.co_attention = CoAttention_TextImage(args)      
        self.co_attention_ = CoAttention_TextImage(args)
        self.co_attention__ = CoAttention_TextImage(args)
        self.co_attention___ = CoAttention_TextImage(args)
        self.co_attention1 = CoAttention_TextDNS(args)       
        self.co_attention1_ = CoAttention_TextDNS(args)
        self.co_attention1__ = CoAttention_TextDNS(args)
        self.co_attention1___ = CoAttention_TextDNS(args)
        self.co_attention2 = CoAttention_ImageDNS(args)
        self.co_attention3 = CoAttention_DNSImage(args)                                                
        self.co_attention2_ = CoAttention_ImageDNS(args)
        self.co_attention3_ = CoAttention_DNSImage(args)
        self.gmf = GMF4(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)

    def forward(self,  img_feature, mask, label_ids,dns_feature,token_ids,token_length,domain_token_ids,domain_token_length):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """
        text_features = self.lstm1(token_ids,token_length,self.args.max_seq_len)
        img_features = self.dim_match(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
        dns_features = self.lstm2(domain_token_ids,domain_token_length,self.args.max_seq_len_dns)

        assert text_features.size(-1) == img_features.size(-1)
        assert text_features.size(-1) == dns_features.size(-1)
        #
        #print('text_features:',text_features.shape,text_features.size(-1))
        #print('img_features:',img_features.shape,img_features.size(-1))
        #att_text_features = self.text_transformer(text_features, mask=None)
        att_img_features = self.img_transformer(img_features, mask=None)

        #att_dns_features = self.dns_transformer(dns_features, mask=None)
        #print('text_features1:',text_features.shape)
        #print('img_features1:',img_features.shape)

        #att_text_features_ = self.co_transformer1(att_text_features,att_img_features, mask=None)
        #att_img_features_ = self.co_transformer2(img_features,text_features, mask=None)

        att_text_features1, att_img_features1 = self.co_attention(text_features, img_features)
        att_text_features2, att_dns_features1 = self.co_attention1(text_features, dns_features)
        #att_img_features2, att_dns_feature2 = self.co_attention(img_features, dns_feature)
        att_text_features3, att_img_features2 = self.co_attention_(text_features, att_img_features)
        
        #att_text_features3, att_img_features2 = self.co_attention_(att_text_features, att_img_features)
        #att_text_features4, att_dns_features2 = self.co_attention1_(att_text_features, att_dns_features)
        #att_img_features4, att_dns_feature4 = self.co_attention(att_img_features, att_dns_feature)

        co_dns_features, co_img_features = self.co_attention2(dns_features, img_features)
        co_dns_features1, co_img_features1 = self.co_attention2_(dns_features, att_img_features)

        co_img_features2, co_dns_features2  = self.co_attention3(dns_features, img_features)
        co_img_features3, co_dns_features3 = self.co_attention3_(dns_features, att_img_features)


        att_text_features5, att_img_features3 = self.co_attention__(text_features, co_img_features2)
        att_text_features6, att_dns_features3 = self.co_attention1__(text_features, co_dns_features)

        att_text_features7, att_img_features4 = self.co_attention___(text_features, co_img_features3)
        att_text_features8, att_dns_features4 = self.co_attention1___(text_features, co_dns_features1)

        #print(att_dns_features3.shape,att_img_features3.shape)
        #print(att_dns_features4.shape,att_img_features4.shape)
        att_text_features = torch.cat((att_text_features1,att_text_features2,att_text_features5,att_text_features6,att_text_features7,att_text_features8), dim=2)
        

        #att_img_features = torch.cat((att_img_features1,att_img_features1,att_img_features2,att_img_features2,att_img_features3,att_img_features3,att_img_features4,att_img_features4), dim=2)
        att_img_features = torch.cat((att_img_features1,att_img_features2,att_img_features3,att_img_features4), dim=2)

        #att_dns_features = torch.cat((att_dns_features1,att_dns_features1,att_dns_features2,att_dns_features2,att_dns_features3,att_dns_features3,att_dns_features4,att_dns_features4), dim=2)
        att_dns_features = torch.cat((att_dns_features1,att_dns_features3,att_dns_features4), dim=2)

        multimodal_features = self.gmf(att_text_features, att_img_features,att_dns_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits


class ACN5(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN5, self).__init__()
        self.lstm = BiLSTM(args, pretrained_word_matrix)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match_dns = nn.Sequential(
            nn.Linear(args.dns_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.co_attention = CoAttention_TextDNS(args)
        self.gmf = GMF(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids, dns_feature, mask, label_ids):
        
        text_features = self.lstm(word_ids, char_ids)
        dns_feature = self.dim_match_dns(dns_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
        assert text_features.size(-1) == dns_feature.size(-1)

        att_text_features, att_dns_features = self.co_attention(text_features, dns_feature)
        multimodal_features = self.gmf(att_text_features, att_dns_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits


class ACN1(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN1, self).__init__()
        self.lstm = BiLSTM(args, pretrained_word_matrix)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        is_cuda = True if torch.cuda.is_available() and not args.no_cuda else False
        self.text_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.img_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
       
        self.co_attention = CoAttention_TextImage(args)                                                           
        self.gmf = GMF1(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids, img_feature, mask, label_ids):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """
        text_features = self.lstm(word_ids, char_ids)
        img_features = self.dim_match(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
        assert text_features.size(-1) == img_features.size(-1)

        #print('text_features:',text_features.shape,text_features.size(-1))
        #print('img_features:',img_features.shape,img_features.size(-1))
        att_text_features1 = self.text_transformer(text_features, mask=None)
        att_img_features1 = self.img_transformer(img_features, mask=None)

        #print('text_features1:',text_features.shape)
        #print('img_features1:',img_features.shape)

        #att_text_features_ = self.co_transformer1(att_text_features,att_img_features, mask=None)
        #att_img_features_ = self.co_transformer2(img_features,text_features, mask=None)

        att_text_features2, att_img_features2 = self.co_attention(text_features, img_features)
        att_text_features1, att_img_features1 = self.co_attention(att_text_features1, att_img_features1)

        
        att_text_features = torch.cat((att_text_features1,att_text_features2), dim=2)
        att_img_features = torch.cat((att_img_features1,att_img_features2), dim=2)

        multimodal_features = self.gmf(att_text_features, att_img_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits

class ACN2(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN2, self).__init__()
        self.lstm = BiLSTM(args, pretrained_word_matrix)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.dim_match_dns = nn.Sequential(
            nn.Linear(args.dns_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        is_cuda = True if torch.cuda.is_available() and not args.no_cuda else False
        self.text_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.dns_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
       
        self.co_attention = CoAttention_TextImage(args)      
        self.co_attention1 = CoAttention_TextDNS(args)                                                       
        self.gmf = GMF2(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids,  mask, label_ids,dns_feature):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """
        text_features = self.lstm(word_ids, char_ids)
        dns_feature = self.dim_match_dns(dns_feature)

        assert text_features.size(-1) == dns_feature.size(-1)
        #
        #print('text_features:',text_features.shape,text_features.size(-1))
        #print('img_features:',img_features.shape,img_features.size(-1))
        att_text_features = self.text_transformer(text_features, mask=None)
        att_dns_feature = self.dns_transformer(dns_feature, mask=None)
        #print('text_features1:',text_features.shape)
        #print('img_features1:',img_features.shape)

        #att_text_features_ = self.co_transformer1(att_text_features,att_img_features, mask=None)
        #att_img_features_ = self.co_transformer2(img_features,text_features, mask=None)

        att_text_features1, att_dns_features1 = self.co_attention1(text_features, dns_feature)
        #att_img_features2, att_dns_feature2 = self.co_attention(img_features, dns_feature)



        att_text_features2, att_dns_features2 = self.co_attention1(att_text_features, att_dns_feature)
        #att_img_features4, att_dns_feature4 = self.co_attention(att_img_features, att_dns_feature)

        att_text_features = torch.cat((att_text_features1,att_text_features2), dim=2)
        

        att_dns_features = torch.cat((att_dns_features1,att_dns_features2), dim=2)
        

        multimodal_features = self.gmf(att_text_features, att_dns_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits

class ACN3(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN3, self).__init__()
        self.lstm = BiLSTM(args, pretrained_word_matrix)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.dim_match_dns = nn.Sequential(
            nn.Linear(args.dns_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        is_cuda = True if torch.cuda.is_available() and not args.no_cuda else False

        self.text_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.img_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.dns_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
       
        self.co_attention = CoAttention_TextImage(args)      
        self.co_attention1 = CoAttention_TextDNS(args)          
        self.co_attention_ = CoAttention_TextImage(args)
        self.co_attention1_ = CoAttention_TextDNS(args)
        self.gmf = GMF3(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids, img_feature, mask, label_ids,dns_feature):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """
        text_features = self.lstm(word_ids, char_ids)
        img_features = self.dim_match(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
        dns_feature = self.dim_match_dns(dns_feature)

        assert text_features.size(-1) == img_features.size(-1)
        assert text_features.size(-1) == dns_feature.size(-1)
        #
        #print('text_features:',text_features.shape,text_features.size(-1))
        #print('img_features:',img_features.shape,img_features.size(-1))
        att_text_features = self.text_transformer(text_features, mask=None)
        att_text_features = text_features
        att_img_features = self.img_transformer(img_features, mask=None)
        #att_dns_feature = self.dns_transformer(dns_feature, mask=None)
        att_dns_feature = dns_feature 
        #print('text_features1:',text_features.shape)
        #print('img_features1:',img_features.shape)

        #att_text_features_ = self.co_transformer1(att_text_features,att_img_features, mask=None)
        #att_img_features_ = self.co_transformer2(img_features,text_features, mask=None)

        att_text_features1, att_img_features1 = self.co_attention(text_features, img_features)
        att_text_features2, att_dns_features1 = self.co_attention1(text_features, dns_feature)
        #att_img_features2, att_dns_feature2 = self.co_attention(img_features, dns_feature)


        att_text_features3, att_img_features2 = self.co_attention_(att_text_features, att_img_features)
        att_text_features4, att_dns_features2 = self.co_attention1_(att_text_features, att_dns_feature)
        #att_img_features4, att_dns_feature4 = self.co_attention(att_img_features, att_dns_feature)

        att_text_features = torch.cat((att_text_features1,att_text_features2,att_text_features3,att_text_features4), dim=2)
        

        att_img_features = torch.cat((att_img_features1,att_img_features1,att_img_features2,att_img_features2), dim=2)
        

        att_dns_features = torch.cat((att_dns_features1,att_dns_features1,att_dns_features2,att_dns_features2), dim=2)
        

        multimodal_features = self.gmf(att_text_features, att_img_features,att_dns_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits

class ACN4(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN4, self).__init__()
        self.lstm = BiLSTM(args, pretrained_word_matrix)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.dim_match_dns = nn.Sequential(
            nn.Linear(args.dns_feat_dim, args.hidden_dim),
            nn.Tanh()
        )
        device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        is_cuda = True if torch.cuda.is_available() and not args.no_cuda else False

        self.text_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.img_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
        self.dns_transformer = TransformerBlock(hidden=args.transformer_hidden_size, 
                                            attn_heads=args.transformer_heads, 
                                            feed_forward_hidden=args.transformer_forward_hidden_size, 
                                            dropout=args.transformer_dropout, 
                                            is_cuda=is_cuda)
       
        self.co_attention = CoAttention_TextImage(args)      
        self.co_attention_ = CoAttention_TextImage(args)
        self.co_attention__ = CoAttention_TextImage(args)
        self.co_attention___ = CoAttention_TextImage(args)
        self.co_attention1 = CoAttention_TextDNS(args)       
        self.co_attention1_ = CoAttention_TextDNS(args)
        self.co_attention1__ = CoAttention_TextDNS(args)
        self.co_attention1___ = CoAttention_TextDNS(args)
        self.co_attention2 = CoAttention_ImageDNS(args)
        self.co_attention3 = CoAttention_DNSImage(args)                                                
        self.co_attention2_ = CoAttention_ImageDNS(args)
        self.co_attention3_ = CoAttention_DNSImage(args)
        self.gmf = GMF4(args)
        self.filtration_gate = FiltrationGate(args)
        self.crf = CRF(num_tags=len(WebsiteProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids, img_feature, mask, label_ids,dns_feature):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_feature: (batch_size, num_img_region(=49), img_feat_dim(=512))
        :param mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len)
        :return:
        """
        text_features = self.lstm(word_ids, char_ids)
        img_features = self.dim_match(img_feature)  # [batch_size, num_img_region(=49), hidden_dim(=200)]
        dns_features = self.dim_match_dns(dns_feature)

        assert text_features.size(-1) == img_features.size(-1)
        assert text_features.size(-1) == dns_features.size(-1)
        #
        #print('text_features:',text_features.shape,text_features.size(-1))
        #print('img_features:',img_features.shape,img_features.size(-1))
        #att_text_features = self.text_transformer(text_features, mask=None)
        att_img_features = self.img_transformer(img_features, mask=None)

        #att_dns_features = self.dns_transformer(dns_features, mask=None)
        #print('text_features1:',text_features.shape)
        #print('img_features1:',img_features.shape)

        #att_text_features_ = self.co_transformer1(att_text_features,att_img_features, mask=None)
        #att_img_features_ = self.co_transformer2(img_features,text_features, mask=None)

        att_text_features1, att_img_features1 = self.co_attention(text_features, img_features)
        att_text_features2, att_dns_features1 = self.co_attention1(text_features, dns_features)
        #att_img_features2, att_dns_feature2 = self.co_attention(img_features, dns_feature)
        att_text_features3, att_img_features2 = self.co_attention_(text_features, att_img_features)
        
        #att_text_features3, att_img_features2 = self.co_attention_(att_text_features, att_img_features)
        #att_text_features4, att_dns_features2 = self.co_attention1_(att_text_features, att_dns_features)
        #att_img_features4, att_dns_feature4 = self.co_attention(att_img_features, att_dns_feature)

        co_dns_features, co_img_features = self.co_attention2(dns_features, img_features)
        co_dns_features1, co_img_features1 = self.co_attention2_(dns_features, att_img_features)

        co_img_features2, co_dns_features2  = self.co_attention3(dns_features, img_features)
        co_img_features3, co_dns_features3 = self.co_attention3_(dns_features, att_img_features)


        att_text_features5, att_img_features3 = self.co_attention__(text_features, co_img_features2)
        att_text_features6, att_dns_features3 = self.co_attention1__(text_features, co_dns_features)

        att_text_features7, att_img_features4 = self.co_attention___(text_features, co_img_features3)
        att_text_features8, att_dns_features4 = self.co_attention1___(text_features, co_dns_features1)

        #print(att_dns_features3.shape,att_img_features3.shape)
        #print(att_dns_features4.shape,att_img_features4.shape)
        att_text_features = torch.cat((att_text_features1,att_text_features2,att_text_features5,att_text_features6,att_text_features7,att_text_features8), dim=2)
        

        #att_img_features = torch.cat((att_img_features1,att_img_features1,att_img_features2,att_img_features2,att_img_features3,att_img_features3,att_img_features4,att_img_features4), dim=2)
        att_img_features = torch.cat((att_img_features1,att_img_features2,att_img_features3,att_img_features4), dim=2)

        #att_dns_features = torch.cat((att_dns_features1,att_dns_features1,att_dns_features2,att_dns_features2,att_dns_features3,att_dns_features3,att_dns_features4,att_dns_features4), dim=2)
        att_dns_features = torch.cat((att_dns_features1,att_dns_features3,att_dns_features4), dim=2)

        multimodal_features = self.gmf(att_text_features, att_img_features,att_dns_features)
        logits = self.filtration_gate(text_features, multimodal_features)

        loss = 0
        if label_ids is not None:
            loss = self.crf(logits, label_ids, mask.byte(), reduction='mean')
            loss = loss * -1  # negative log likelihood

        return loss, logits
