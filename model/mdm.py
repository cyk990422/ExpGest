import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from local_attention import LocalAttention
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel
'''
import torch 
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
'''

#语义联合
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="../mydiffusion_zeggs/1dbc166cf8765166998eff31ade2eb64c8a40076", output_dim=256):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, texts):
        # Tokenize 文本
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()

        # 使用 BERT 编码文本
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_output.last_hidden_state

        # 取第一个 token（[CLS] 标记）的隐藏状态作为文本的表示
        cls_token = hidden_states[:, 0, :]

        # 调整输出维度
        features = self.linear(cls_token)

        return features

class SemGes_Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.Textencoder_body=TextEncoder()
        self.Textencoder_hand=TextEncoder()


    def forward(self,text):
        text_latent_body=self.Textencoder_body(text)#bs,11,256
        text_latent_hand=self.Textencoder_hand(text)
        #print("text_latent_body shape:",text_latent_body.shape)
        # 使用平均池化提取手势隐变量
        return text_latent_body.unsqueeze(1),text_latent_hand.unsqueeze(1)




#情感联合
class WavEncoder_emo(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_map = nn.Linear(1024, 256)

    def forward(self, rep):
        rep = self.audio_feature_map(rep)
        return rep  


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, query, key):
        # query: (batch_size, 64)
        # key: (batch_size, 88, 64)
        query = query.unsqueeze(1)  # (batch_size, 1, 64)
        key = key.transpose(0, 1)  # (88, batch_size, 64)
        query = query.transpose(0, 1)  # (1, batch_size, 64)
        attn_output, _ = self.attention(query, key, key)  # (1, batch_size, 64)
        attn_output = attn_output.transpose(0, 1).squeeze(1)  # (batch_size, 64)
        return attn_output

class Emorhy_Ges(nn.Module):
    def __init__(self):
        super().__init__()
        self.WavEncoder_body=WavEncoder_emo()
        self.WavEncoder_hand=WavEncoder_emo()
        self.embed_emotion_body = nn.Linear(8, 256)
        self.embed_emotion_hand = nn.Linear(8, 256)
        self.avg_pool = nn.AvgPool1d(11)

        self.cross_attention_body=CrossAttention(256, 8)
        self.cross_attention_hand=CrossAttention(256, 8)

    def forward(self,wavlm,style):
        emo_feat_body=self.embed_emotion_body(style)#bs,256
        emo_feat_hand=self.embed_emotion_hand(style)

        wavlm_body= self.WavEncoder_body(wavlm)#bs,88,64
        wavlm_hand= self.WavEncoder_hand(wavlm)
        
        attn_output_body = self.cross_attention_body(emo_feat_body, wavlm_body)
        attn_output_hand = self.cross_attention_hand(emo_feat_hand, wavlm_hand)


        return attn_output_body,attn_output_hand





class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, audio_feat='', n_seed=1, cond_mode='', **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0

        self.audio_feat = audio_feat
        if audio_feat == 'wav encoder':
            self.audio_feat_dim = 32
        elif audio_feat == 'mfcc':
            self.audio_feat_dim = 13
        elif self.audio_feat == 'wavlm':
            print('USE WAVLM')
            self.audio_feat_dim = 64        # Linear 1024 -> 64
            self.WavEncoder = WavEncoder()
            #self.WavEncoder = AudioEncoder()

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.cond_mode = cond_mode
        self.num_head = 8

        #语义联合
        self.semges=SemGes_Dis()
        checkpoint=torch.load("../mydiffusion_zeggs/audio_only_model/joint_sem_ges.bin")
        self.semges.load_state_dict(checkpoint['state_dict'], strict=False)
        self.semges.eval()
        self.semges.cuda()


        if 'style2' not in self.cond_mode:
            self.input_process_body = InputProcess(self.data_rep, 451 + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)
            self.input_process_hand = InputProcess(self.data_rep, 690 + self.audio_feat_dim + self.gru_emb_dim, self.latent_dim)

        if self.arch == 'mytrans_enc':
            print("MY TRANS_ENC init")
            from mytransformer import TransformerEncoderLayer, TransformerEncoder

            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(1536, self.latent_dim)

            seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer_body = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_body = nn.TransformerEncoder(seqTransEncoderLayer_body,
                                                         num_layers=self.num_layers)

            seqTransEncoderLayer_body1 = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_body1 = nn.TransformerEncoder(seqTransEncoderLayer_body1,
                                                         num_layers=self.num_layers)                                              
            
            seqTransEncoderLayer_hand = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_hand = nn.TransformerEncoder(seqTransEncoderLayer_hand,
                                                         num_layers=self.num_layers)

            seqTransEncoderLayer_hand1 = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder_hand1 = nn.TransformerEncoder(seqTransEncoderLayer_hand1,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=False)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        #self.embed_timestep_body = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        #self.embed_timestep_hand = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        if 'style1' in self.cond_mode:
            print('EMBED STYLE BEGIN TOKEN')
            if self.n_seed != 0:
                self.style_dim = 64
                self.embed_style = nn.Linear(8, self.style_dim)
                #self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim - self.style_dim)
                self.embed_text_body = nn.Linear(451 * n_seed, self.latent_dim - self.style_dim)
                self.embed_text_hand = nn.Linear(690 * n_seed, self.latent_dim - self.style_dim)
            else:
                self.style_dim = self.latent_dim
                self.embed_style = nn.Linear(8, self.style_dim)

        elif 'style2' in self.cond_mode:
            print('EMBED STYLE ALL FRAMES')
            self.style_dim = 64
            self.embed_style = nn.Linear(8, self.style_dim)
            self.input_process = InputProcess(self.data_rep, self.input_feats + self.audio_feat_dim + self.gru_emb_dim + self.style_dim,
                                              self.latent_dim)
            if self.n_seed != 0:
                self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim)
        elif self.n_seed != 0:
            self.embed_text = nn.Linear(self.njoints * n_seed, self.latent_dim)

        self.output_process_body = OutputProcess(self.data_rep, 451, self.latent_dim, 451,
                                            self.nfeats)
        self.output_process_hand = OutputProcess(self.data_rep, 690, self.latent_dim, 690,
                                            self.nfeats)

        if 'cross_local_attention' in self.cond_mode:
            self.rel_pos_body = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.rel_pos_hand = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            #self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)
            self.input_process_body = InputProcess(self.data_rep, 451  + self.gru_emb_dim, self.latent_dim)
            self.input_process_hand = InputProcess(self.data_rep, 690  + self.gru_emb_dim, self.latent_dim)
            self.cross_local_attention_body = LocalAttention(
                dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=11,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.cross_local_attention_hand = LocalAttention(
                dim=32,  # dimension of each head (you need to pass this in for relative positional encoding)
                window_size=11,  # window size. 512 is optimal, but 256 or 128 yields good enough results
                causal=True,  # auto-regressive or not
                look_backward=1,  # each window looks at the window before
                look_forward=0,     # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
                dropout=0.1,  # post-attention dropout
                exact_windowsize=False
                # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
            )
            self.input_process2_body = nn.Linear(self.latent_dim * 3 + self.audio_feat_dim, self.latent_dim)
            self.input_process2_hand = nn.Linear(self.latent_dim * 3 + self.audio_feat_dim, self.latent_dim)

        if 'cross_local_attention2' in self.cond_mode:
            print('Cross Local Attention2')
            self.selfAttention = LinearTemporalCrossAttention(seq_len=0, latent_dim=256, text_latent_dim=256, num_head=8, dropout=0.1, time_embed_dim=0)

        if 'cross_local_attention3' in self.cond_mode:
            print('Cross Local Attention3')

        if 'cross_local_attention4' in self.cond_mode:
            print('Cross Local Attention4')

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x_body, x_hand, timesteps,hy_flag=False, y=None,uncond_info=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper  x = torch.randn(2, 1140, 1, 88)
        timesteps: [batch_size] (int)
        seed: [batch_size, njoints, nfeats]
        """
        #print("model!!!!")
        #bs, njoints, nfeats, nframes = x.shape      # 64, 251, 1, 196
        #print("x_body.shape:",x_body.shape)#300,451,1,88
        #bs,nframes,njoints=x_body.shape#300,88,256

        if hy_flag==False:
            bs, njoints_body, nfeats_body, nframes = x_body.shape
            bs, njoints_hand, nfeats_hand, nframes = x_hand.shape

            emb_t=self.embed_timestep(timesteps)
            

            #emb_t_body = self.embed_timestep_body(timesteps)  # [1, bs, d], (1, 2, 256)
            #emb_t_hand = self.embed_timestep_hand(timesteps)

            #force_mask = y.get('uncond', False)  # False
            force_mask=uncond_info

            
            text=y['wav_text']
            #print("text:",text)
            #训练时候开启
            '''
            with torch.no_grad():
                text_latent_body,text_latent_hand=self.semges(text)
                #attn_output_body,attn_output_hand=self.emorhy(y['audio'],y['style'])
            '''
            #采样时候开启
            text_latent_body = y['text_latent_body']
            text_latent_hand = y['text_latent_hand']
            
            if 'style1' in self.cond_mode:
                embed_style = self.mask_cond(self.embed_style(y['style']), force_mask=force_mask)       # (bs, 64)
                if self.n_seed != 0:
                    embed_text_body = self.embed_text_body(self.mask_cond(y['seed_body'].squeeze(2).reshape(bs, -1), force_mask=force_mask))       # (bs, 256-64)
                    embed_text_hand = self.embed_text_hand(self.mask_cond(y['seed_hand'].squeeze(2).reshape(bs, -1), force_mask=force_mask))       # (bs, 256-64)
                    emb_1_body = torch.cat((embed_style, embed_text_body), dim=1)
                    emb_1_hand = torch.cat((embed_style, embed_text_hand), dim=1)
                else:
                    emb_1_body = embed_style
                    emb_1_hand = embed_style
            elif self.n_seed != 0:
                emb_1_body = self.embed_text_body(self.mask_cond(y['seed_body'].squeeze(2).reshape(bs, -1), force_mask=force_mask))     # z_tk
                emb_1_hand = self.embed_text_hand(self.mask_cond(y['seed_hand'].squeeze(2).reshape(bs, -1), force_mask=force_mask))     # z_tk

            if self.audio_feat == 'wavlm':
                enc_text = self.WavEncoder(y['audio']).permute(1, 0, 2)
            else:
                enc_text = y['audio']

            if 'cross_local_attention' in self.cond_mode:
                if 'cross_local_attention3' in self.cond_mode:
                    #x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [2, 135, 1, 240]  
                    # self-attention
                    #print("x shape:",x.shape)#x shape: torch.Size([896, 1141, 1, 88])
                    #x_ = self.input_process(x)  # [2, 135, 1, 240] -> [240, 2, 256]
                    #print("x_ shape:",x_.shape)#x_ shape: torch.Size([88, 896, 256])
                    x_body = x_body.reshape(bs, njoints_body * nfeats_body, 1, nframes)
                    x_hand = x_hand.reshape(bs, njoints_hand * nfeats_hand, 1, nframes)
                    x_body_=self.input_process_body(x_body)
                    x_hand_=self.input_process_hand(x_hand)
                    #print("x_body_ shape:",x_body_.shape)
                    # local-cross-attention
                    packed_shape = [torch.Size([bs, self.num_head])]
                    #xseq = torch.cat((x_, enc_text), axis=2)  # [bs, d+joints*feat, 1, #frames], (240, 2, 32)

                    #print("x_body_ shape:",x_body_.shape,"    enc_text shape:",enc_text.shape)
                    xseq_body = torch.cat((x_body_, enc_text), axis=2)
                    xseq_hand = torch.cat((x_hand_, enc_text), axis=2)
                    #print("xseq +audio shape:",xseq.shape)
                    # all frames
                    #embed_style_2 = (emb_1 + emb_t).repeat(nframes, 1, 1)  # (bs, 64) -> (len, bs, 64)
                    embed_style_2_body = (emb_1_body + emb_t).repeat(nframes, 1, 1)
                    embed_style_2_hand = (emb_1_hand + emb_t).repeat(nframes, 1, 1)

                    #嵌入局部语义
                    text_latent_body_local=text_latent_body.squeeze(1).repeat(nframes, 1, 1)
                    text_latent_hand_local=text_latent_hand.squeeze(1).repeat(nframes, 1, 1)
                    #text_latent_hand_local=text_latent_hand.expand(-1,11,256)
                    embed_style_2_body=torch.cat((embed_style_2_body,text_latent_body_local),axis=2)
                    embed_style_2_hand=torch.cat((embed_style_2_hand,text_latent_hand_local),axis=2)
                    '''
                    #嵌入情感风格
                    attn_output_body_local=attn_output_body.squeeze(1).repeat(nframes, 1, 1)
                    attn_output_hand_local=attn_output_hand.squeeze(1).repeat(nframes, 1, 1)
                    embed_style_2_body=torch.cat((embed_style_2_body,attn_output_body_local),axis=2)
                    embed_style_2_hand=torch.cat((embed_style_2_hand,attn_output_hand_local),axis=2)
                    '''

                    
                    xseq_body = torch.cat((embed_style_2_body, xseq_body), axis=2)  # (seq, bs, dim)
                    xseq_hand = torch.cat((embed_style_2_hand, xseq_hand), axis=2)
                    #print("xseq +embed_style_2 shape:",xseq.shape)
                    #xseq = self.input_process2(xseq)
                    #print("xseq_body shape 2:",xseq_body.shape)
                    xseq_body = self.input_process2_body(xseq_body)
                    xseq_hand = self.input_process2_hand(xseq_hand)#88,300,256
                    xseq_body = xseq_body.permute(1, 0, 2)  # (bs, len, dim)
                    xseq_body = xseq_body.view(bs, nframes, self.num_head, -1)
                    xseq_body = xseq_body.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                    xseq_body = xseq_body.reshape(bs * self.num_head, nframes, -1)
                    
                    xseq_hand = xseq_hand.permute(1, 0, 2)  # (bs, len, dim)
                    xseq_hand = xseq_hand.view(bs, nframes, self.num_head, -1)
                    xseq_hand = xseq_hand.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                    xseq_hand = xseq_hand.reshape(bs * self.num_head, nframes, -1)
                

                    pos_emb_body = self.rel_pos_body(xseq_body)  # (89, 32)
                    pos_emb_hand = self.rel_pos_hand(xseq_hand)


                    xseq_body, _ = apply_rotary_pos_emb(xseq_body, xseq_body, pos_emb_body)
                    xseq_rpe_body = xseq_body.reshape(bs, self.num_head, nframes, -1)
                    xseq_hand, _ = apply_rotary_pos_emb(xseq_hand, xseq_hand, pos_emb_hand)
                    xseq_rpe_hand = xseq_hand.reshape(bs, self.num_head, nframes, -1)
                    '''
                    #vit
                    xseq_body = xseq_rpe_body.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                    xseq_body = xseq_body.view(bs, nframes, -1)
                    xseq_body = xseq_body.permute(1, 0, 2)

                    xseq_hand = xseq_rpe_hand.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                    xseq_hand = xseq_hand.view(bs, nframes, -1)
                    xseq_hand = xseq_hand.permute(1, 0, 2) #11,300,256
                    xseq_body = self.seqTransEncoder_body1(xseq_body)
                    xseq_hand = self.seqTransEncoder_hand1(xseq_hand)
                    '''

                    #cross-local
                    
                    xseq_body = self.cross_local_attention_body(xseq_body, xseq_body, xseq_body, packed_shape=packed_shape,
                                                    mask=y['mask_local'])  # (2, 8, 2048, 64)
                    xseq_hand = self.cross_local_attention_hand(xseq_hand, xseq_hand, xseq_hand, packed_shape=packed_shape,
                                                    mask=y['mask_local'])  # (2, 8, 2048, 64)
                    

                    xseq_body = xseq_body.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                    xseq_body = xseq_body.reshape(bs, nframes, -1)
                    xseq_body = xseq_body.permute(1, 0, 2)
                    xseq_hand = xseq_hand.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                    xseq_hand = xseq_hand.reshape(bs, nframes, -1)
                    xseq_hand = xseq_hand.permute(1, 0, 2)
                    

                    #print("xseq_body shape:",xseq_body.shape,"      emb_1_body shape:",emb_1_body.shape,"     text_latent_hand_global shape:",text_latent_hand_global.shape,"   emb_t shape:",emb_t.shape)
                    #xseq_body = torch.cat((emb_1_body + emb_t,text_latent_body.permute(1, 0, 2),attn_output_body.unsqueeze(1).permute(1, 0, 2), xseq_body), axis=0)
                    #xseq_hand = torch.cat((emb_1_hand + emb_t,text_latent_hand.permute(1, 0, 2),attn_output_hand.unsqueeze(1).permute(1, 0, 2), xseq_hand), axis=0)

                    xseq_body = torch.cat((emb_1_body + emb_t,text_latent_body.permute(1, 0, 2), xseq_body), axis=0)
                    xseq_hand = torch.cat((emb_1_hand + emb_t,text_latent_hand.permute(1, 0, 2), xseq_hand), axis=0)
                    

                    #xseq_body = torch.cat((emb_1_body + emb_t, xseq_body), axis=0)
                    #xseq_hand = torch.cat((emb_1_hand + emb_t, xseq_hand), axis=0)

                    xseq_body = xseq_body.permute(1, 0, 2)  # (bs, len, dim)
                    xseq_body = xseq_body.view(bs, nframes + 2, self.num_head, -1)
                    xseq_body = xseq_body.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                    xseq_body = xseq_body.reshape(bs * self.num_head, nframes + 2, -1)
                    xseq_hand = xseq_hand.permute(1, 0, 2)  # (bs, len, dim)
                    xseq_hand = xseq_hand.view(bs, nframes + 2, self.num_head, -1)
                    xseq_hand = xseq_hand.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                    xseq_hand = xseq_hand.reshape(bs * self.num_head, nframes + 2, -1)

                    pos_emb_body = self.rel_pos_body(xseq_body)
                    pos_emb_hand = self.rel_pos_hand(xseq_hand)
                    xseq_body, _ = apply_rotary_pos_emb(xseq_body, xseq_body, pos_emb_body)
                    xseq_rpe_body = xseq_body.reshape(bs, self.num_head, nframes + 2, -1)
                    xseq_hand, _ = apply_rotary_pos_emb(xseq_hand, xseq_hand, pos_emb_hand)
                    xseq_rpe_hand = xseq_hand.reshape(bs, self.num_head, nframes + 2, -1)
                    
                    xseq_body = xseq_rpe_body.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                    xseq_body = xseq_body.view(bs, nframes + 2, -1)
                    xseq_body = xseq_body.permute(1, 0, 2)

                    xseq_hand = xseq_rpe_hand.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                    xseq_hand = xseq_hand.view(bs, nframes + 2, -1)
                    xseq_hand = xseq_hand.permute(1, 0, 2)

                    '''
                    xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
                                                    mask=y['mask_local'])  # (2, 8, 2048, 64)
                    '''
                    '''
                    xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                    xseq = xseq.reshape(bs, nframes, -1)
                    xseq = xseq.permute(1, 0, 2)
                    print("xseq +cross_local_attention shape:",xseq.shape)
                    '''

                    #xseq = torch.cat((emb_1 + emb_t, xseq), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                    #print("xseq +att shape:",xseq.shape)
                    '''
                    xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
                    xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
                    xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                    xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
                    '''
                    #pos_emb = self.rel_pos(xseq)  # (89, 32)
                    #xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                    #xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
                    '''
                    xseq = xseq_rpe.permute(0, 2, 1, 3)  # [seqlen+1, bs, d]
                    xseq = xseq.view(bs, nframes + 1, -1)
                    xseq = xseq.permute(1, 0, 2)
                    '''
                    if 'cross_local_attention2' in self.cond_mode:
                        xseq = (self.selfAttention(xseq).permute(1, 0, 2))[2:]
                    else:
                        output_body = self.seqTransEncoder_body(xseq_body)[2:]
                        output_hand = self.seqTransEncoder_hand(xseq_hand)[2:]
                        
                    #print("output shape:",output.shape)

                elif 'cross_local_attention5' in self.cond_mode:
                    x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [2, 135, 1, 240]
                    # self-attention
                    x_ = self.input_process(x)  # [2, 135, 1, 240] -> [240, 2, 256]

                    # local-cross-attention
                    packed_shape = [torch.Size([bs, self.num_head])]
                    xseq = torch.cat((x_, enc_text), axis=2)  # [bs, d+joints*feat, 1, #frames], (240, 2, 32)
                    # all frames
                    embed_style_2 = (emb_1 + emb_t).repeat(nframes, 1, 1)  # (bs, 64) -> (len, bs, 64)
                    xseq = torch.cat((embed_style_2, xseq), axis=2)  # (seq, bs, dim)
                    xseq = self.input_process2(xseq)
                    xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
                    xseq = xseq.view(bs, nframes, self.num_head, -1)
                    xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                    xseq = xseq.reshape(bs * self.num_head, nframes, -1)
                    pos_emb = self.rel_pos(xseq)  # (89, 32)
                    xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                    xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape,
                                                    mask=y['mask_local'])  # (2, 8, 2048, 64)
                    xseq = xseq.permute(0, 2, 1, 3)  # (bs, len, 8, 64)
                    xseq = xseq.reshape(bs, nframes, -1)
                    output = xseq.permute(1, 0, 2)

                else:
                    x = x.reshape(bs, njoints*nfeats, 1, nframes)      # [2, 135, 1, 240]
                    # self-attention
                    x_ = self.input_process(x)       # [2, 135, 1, 240] -> [240, 2, 256]
                    xseq = torch.cat((emb_1 + emb_t, x_), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)
                    xseq = xseq.permute(1, 0, 2)        # (bs, len, dim)
                    xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
                    xseq = xseq.permute(0, 2, 1, 3)     # Need (2, 8, 2048, 64)
                    xseq = xseq.reshape(bs*self.num_head, nframes + 1, -1)
                    pos_emb = self.rel_pos(xseq)        #  (89, 32)
                    xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                    xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
                    xseq = xseq_rpe.permute(0, 2, 1, 3)       # [seqlen+1, bs, d]
                    xseq = xseq.view(bs, nframes + 1, -1)
                    xseq = xseq.permute(1, 0, 2)
                    if 'cross_local_attention2' in self.cond_mode:
                        xseq = (self.selfAttention(xseq).permute(1, 0, 2))[1:]
                    else:
                        xseq = self.seqTransEncoder(xseq)[1:]

                    # local-cross-attention
                    packed_shape = [torch.Size([bs, self.num_head])]
                    xseq = torch.cat((xseq, enc_text), axis=2)  #[bs, d+joints*feat, 1, #frames], (240, 2, 32)
                    # all frames
                    embed_style_2 = (emb_1 + emb_t).repeat(nframes, 1, 1)       # (bs, 64) -> (len, bs, 64)
                    xseq = torch.cat((embed_style_2, xseq), axis=2)     # (seq, bs, dim)
                    xseq = self.input_process2(xseq)
                    xseq = xseq.permute(1, 0, 2)  # (bs, len, dim)
                    xseq = xseq.view(bs, nframes, self.num_head, -1)
                    xseq = xseq.permute(0, 2, 1, 3)  # Need (2, 8, 2048, 64)
                    xseq = xseq.reshape(bs * self.num_head, nframes, -1)
                    pos_emb = self.rel_pos(xseq)  # (89, 32)
                    xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                    xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=packed_shape, mask=y['mask_local'])     # (2, 8, 2048, 64)
                    xseq = xseq.permute(0, 2, 1, 3)     # (bs, len, 8, 64)
                    xseq = xseq.reshape(bs, nframes, -1)
                    output = xseq.permute(1, 0, 2)

            else:
                if self.arch == 'trans_enc' or self.arch == 'trans_dec' or self.arch == 'conformers_enc' or self.arch == 'mytrans_enc':
                    x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)      # [2, 135, 1, 240]
                    enc_text_gru = enc_text.permute(1, 2, 0)  # (240, 2, 32) -> (2, 32, 240)
                    enc_text_gru = enc_text_gru.reshape(bs, self.audio_feat_dim, 1, nframes)
                    x = torch.cat((x_reshaped, enc_text_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]
                    if 'style2' in self.cond_mode:
                        embed_style = self.mask_cond(self.embed_style(y['style']), force_mask=force_mask).repeat(nframes, 1, 1)  # (#frames, bs, 64)
                        embed_style = embed_style.unsqueeze(2)
                        embed_style = embed_style.permute(1, 3, 2, 0)
                        x = torch.cat((x, embed_style), axis=1)  # [bs, d+joints*feat, 1, #frames]

                if self.arch == 'gru':
                    x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)      # [2, 135, 1, 240]
                    emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]

                    enc_text_gru = enc_text.permute(1, 2, 0)        # (240, 2, 32) -> (2, 32, 240)
                    enc_text_gru = enc_text_gru.reshape(bs, self.audio_feat_dim, 1, nframes)

                    emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
                    emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
                    x = torch.cat((x_reshaped, emb_gru, enc_text_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

                x = self.input_process(x)       # [2, 135, 1, 240] -> [240, 2, 224]

                if self.arch == 'trans_enc':
                    # adding the timestep embed
                    # x = torch.cat((x, enc_text), axis=2)        # [[240, 2, 224], (240, 2, 32)] -> (240, 2, 256)
                    xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)

                    xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                    output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]      # -> (240, 2, 256)

                elif self.arch == 'trans_dec':
                    if self.emb_trans_dec:
                        xseq = torch.cat((emb, x), axis=0)
                    else:
                        xseq = x
                    xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                    if self.emb_trans_dec:
                        output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
                    else:
                        output = self.seqTransDecoder(tgt=xseq, memory=emb)

                elif self.arch == 'gru':
                    xseq = x
                    xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
                    # pdb.set_trace()
                    output, _ = self.gru(xseq)

                elif self.arch == 'mytrans_enc':
                    # adding the timestep embed
                    # x = torch.cat((x, enc_text), axis=2)        # [[240, 2, 224], (240, 2, 32)] -> (240, 2, 256)
                    xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]     # [(1, 2, 256), (240, 2, 256)] -> (241, 2, 256)

                    sinusoidal_pos = self.embed_positions(xseq.shape[0], 0)[None, None, :, :].chunk(2, dim=-1)
                    xseq = self.apply_rotary(xseq.permute(1, 0, 2), sinusoidal_pos).squeeze(0).permute(1, 0, 2)

                    output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]      # -> (240, 2, 256)
            output_body = self.output_process_body(output_body)
            output_hand = self.output_process_hand(output_hand)
            return output_body.to("cuda"),output_hand.to("cuda")
        
    
            

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, xf=None, emb=None):
        """
        x: B, T, D      , [240, 2, 256]
        xf: B, N, L     , [1, 2, 256]
        """
        x = x.permute(1, 0, 2)
        # xf = xf.permute(1, 0, 2)
        B, T, D = x.shape
        # N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(x)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        return y


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_map = nn.Linear(1024, 64)

    def forward(self, rep):
        rep = self.audio_feature_map(rep)
        return rep

class AudioEncoder(nn.Module):

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv1d_time = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=8, stride=8)
        self.conv1x1_channel = nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)        # 转置为 (batch_size, 1024, 88)
        x = self.conv1d_time(x)      # (batch_size, 1024, 11)
        x = self.conv1x1_channel(x)  # (batch_size, 64, 11)
        x = x.transpose(1, 2)        # 转置为 (batch_size, 11, 64)
        return x

if __name__ == '__main__':
    '''
    cd ./main/model
    python mdm.py
    '''
    n_frames = 240

    n_seed = 8

    model = MDM(modeltype='', njoints=1140, nfeats=1, cond_mode = 'cross_local_attention5_style1', action_emb='tensor', audio_feat='mfcc',
                arch='mytrans_enc', latent_dim=256, n_seed=n_seed, cond_mask_prob=0.1)

    x = torch.randn(2, 1140, 1, 88)
    t = torch.tensor([12, 85])

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1)     # [..., n_seed:]
    model_kwargs_['y']['audio'] = torch.randn(2, 88, 13).permute(1, 0, 2)       # [n_seed:, ...]
    model_kwargs_['y']['style'] = torch.randn(2, 6)
    model_kwargs_['y']['mask_local'] = torch.ones(2, 88).bool()
    model_kwargs_['y']['seed'] = x[..., 0:n_seed]
    y = model(x, t, model_kwargs_['y'])
    print(y.shape)
