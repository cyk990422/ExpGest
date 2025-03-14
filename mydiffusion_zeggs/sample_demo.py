import sys
[sys.path.append(i) for i in ['.', '..', '../process', '../model', '../ubisoft-laforge-ZeroEGGS-main', '../ubisoft-laforge-ZeroEGGS-main/ZEGGS']]
from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
import subprocess
import os
from datetime import datetime
from mfcc import MFCC
import librosa
import numpy as np
import yaml
from pprint import pprint
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from process_beats_bvh_2_smplx_20fps import pose2bvh, quat, pose2json
from process_beats_npz_2_smplx_20fps import pose2json2, pose2json_directly
import argparse
import torch.nn as nn
from mld.mld.models.architectures.mld_vae import MldVae
from scipy.signal import savgol_filter

emotion2onehot = {
'netural':[1, 0, 0, 0, 0, 0, 0, 0],
'happiness':[0, 1, 0, 0, 0, 0, 0, 0],
'anger':[0, 0, 1, 0, 0, 0, 0, 0],
'sadness':[0, 0, 0, 1, 0, 0, 0, 0],
'contempt':[0, 0, 0, 0, 1, 0, 0, 0],
'surprise':[0, 0, 0, 0, 0, 1, 0, 0],
'fear':[0, 0, 0, 0, 0, 0, 1, 0],
'disgust':[0, 0, 0, 0, 0, 0, 0, 1],
}
body_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,36,37,38,39,63,64,65,66,67,68,69,70,71,72,73,74]#29
hand_index=[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]#46

import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor


from transformers import AutoTokenizer, AutoModel

from torch.nn import functional as F
#旋转表示转换
def rot6d_to_rotmat(x):
    x = x.reshape(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def batch_rot2aa(Rs):
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

#语义-手势联合嵌入空间
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="./1dbc166cf8765166998eff31ade2eb64c8a40076", output_dim=256):
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





def init_body_vae_model(args, _device):
    print("init BODY_VAE model")
    GesVAE = MldVae(ablation= {'VAE_TYPE': 'actor', 'VAE_ARCH': 'encoder_decoder', 'PE_TYPE': 'mld', 'DIFF_PE_TYPE': 'mld', 'SKIP_CONNECT': True, 'MLP_DIST': False, 'IS_DIST': False, 'PREDICT_EPSILON': True} 
    ,nfeats=451,latent_dim=[1, 256],ff_size=1024,num_layers=9,num_heads=4,
    dropout=0.1,arch='encoder_decoder',normalize_before=False,activation="gelu",
    position_embedding='learned').to(_device)
    return GesVAE

def init_hand_vae_model(args, _device):
    print("init HAND_VAE model")
    GesVAE = MldVae(ablation= {'VAE_TYPE': 'actor', 'VAE_ARCH': 'encoder_decoder', 'PE_TYPE': 'mld', 'DIFF_PE_TYPE': 'mld', 'SKIP_CONNECT': True, 'MLP_DIST': False, 'IS_DIST': False, 'PREDICT_EPSILON': True} 
    ,nfeats=690,latent_dim=[1, 256],ff_size=1024,num_layers=9,num_heads=4,
    dropout=0.1,arch='encoder_decoder',normalize_before=False,activation="gelu",
    position_embedding='learned').to(_device)
    return GesVAE


class HyVAE(nn.Module):
    def __init__(self,body_vae_model,hand_vae_model):
        super().__init__()
        self.body_vae=body_vae_model
        self.hand_vae=hand_vae_model

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




class EmoCls(nn.Module):
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

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.cond_mode = cond_mode
        self.num_head = 8

        
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.diffusion = create_model_and_diffusion_1(args)


        self.body_vae_model=init_body_vae_model(args, "cuda")
        self.hand_vae_model=init_hand_vae_model(args, "cuda")
        self.hyvae=HyVAE(self.body_vae_model,self.hand_vae_model)
        self.hyvae.eval()
        self.hyvae.cuda()

        self.emo_pred_body=nn.Linear(256, 8)
        self.emo_pred_hand=nn.Linear(256, 8)
       
    def classify(self, x_body, x_hand,t,flag, y=None,uncond_info=False):
        # Noisy Motion
        emb_t=self.embed_timestep(t)


        self.lengths = torch.tensor([len(feature) for feature in x_body])
        self.lengths=self.lengths.cuda()
        if flag=='body':
            body_latent,_ = self.hyvae.body_vae.encode(x_body,self.lengths)
            body_latent=body_latent.permute(1,0,2)
            pred_emotion_body=self.emo_pred_body(body_latent.squeeze(dim=1))
            pred_emotion_hand=None
        elif flag=='hands':
            hand_latent,_ = self.hyvae.hand_vae.encode(x_hand,self.lengths)
        
            hand_latent=hand_latent.permute(1,0,2)#bs,1,256

        
            pred_emotion_hand=self.emo_pred_hand(hand_latent.squeeze(dim=1))#bs,8
            pred_emotion_body=None

        return pred_emotion_body,pred_emotion_hand




class Wave2Vec2Inference:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if use_lm_if_possible:            
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits            

        if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
            transcription = \
                self.processor.decode(logits[0].cpu().numpy(),                                      
                                      hotwords=self.hotwords,
                                      #hotword_weight=self.hotword_weight,  
                                      output_word_offsets=True,                                      
                                   )                             
            confidence = transcription.lm_score / len(transcription.text.split(" "))
            transcription = transcription.text       
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits,predicted_ids)

        return transcription, confidence   

    def confidence_score(self, logits, predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)


asr = Wave2Vec2Inference("./54074b1c16f4de6a5ad59affb4caa8f2ea03a119")


def wavlm_init(device=torch.device('cuda:2')):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    wavlm_model_path = './WavLM/WavLM-Large.pt'
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))     # load the pre-trained checkpoints
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def wav2wavlm(model, args, wav_input_16khz, device=torch.device('cuda:2')):
    wav_input_16khz = wav_input_16khz.to(device)
    rep = model.extract_features(wav_input_16khz)[0]
    rep = F.interpolate(rep.transpose(1, 2), size=args.n_poses, align_corners=True, mode='linear').transpose(1, 2)
    return rep


def create_model_and_diffusion(args):
    if args.mode == 'audio_only':
        model = MDM(modeltype='', njoints=1141, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
                    glob_rot=True, cond_mode = 'cross_local_attention3_style1', clip_version = 'ViT-B/32', action_emb = 'tensor', audio_feat=args.audio_feat,
                    arch='trans_enc', latent_dim=256, n_seed=8)        # trans_enc, trans_dec, gru, mytrans_enc
        diffusion = create_gaussian_diffusion()
    else:
        model = MDM(modeltype='', njoints=994, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
                glob_rot=True, cond_mode = 'cross_local_attention3_style1', clip_version = 'ViT-B/32', action_emb = 'tensor', audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=512, n_seed=8)        # trans_enc, trans_dec, gru, mytrans_enc
        diffusion = create_gaussian_diffusion()
    return model, diffusion


def create_model_and_diffusion_1(args):
    diffusion = create_gaussian_diffusion()
    return diffusion


def inference(args, wavlm_model, audio, sample_fn, model, n_frames=0, smoothing=False, SG_filter=False, minibatch=False, skip_timesteps=0, n_seed=8, style=None, seed=123456,use_emo_classifier=0):
    torch.manual_seed(seed) 
    total_recon_loss = 0
    if n_frames == 0:
        n_frames = audio.shape[0] * 20 // 16000
    last_n_frames = n_frames
    if minibatch:
        stride_poses = args.n_poses - n_seed
        if n_frames < stride_poses:
            num_subdivision = 1
        else:
            #num_subdivision = math.floor(n_frames / stride_poses)
            num_subdivision = math.ceil(n_frames / stride_poses)
            n_frames = num_subdivision * stride_poses
            print(
                '{}, {}, {}'.format(num_subdivision, stride_poses, n_frames))
    audio = audio[:n_frames * int(16000 / 20)]
    audio_padding = np.zeros((n_frames * int(16000 / 20)-audio.shape[0]))
    audio = np.concatenate((audio,audio_padding),axis = 0)

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    # classifier init  噪声情感分类器
    classifier=EmoCls(modeltype='', njoints=1141, nfeats=1, cond_mode = 'cross_local_attention3_style1', action_emb = 'tensor',
                arch='trans_enc', latent_dim=256, n_seed=8, cond_mask_prob=0.1)
    checkpoint = torch.load("./audio_only_model/emo_classifier.bin", map_location="cuda")
    
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith("module."):
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    classifier.load_state_dict(new_state_dict)
    
    classifier.eval()
    classifier.cuda()

    # 语义对齐模型
    semges=SemGes_Dis()
    #checkpoint=torch.load("./audio_only_model/joint_sem_ges.bin")
    checkpoint=torch.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/DiffuseStyleGesture/main/mydiffusion_zeggs/output/train_beat_vae_sememb/vae_checkpoint_1200.bin")
    #checkpoint=torch.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/DiffuseStyleGesture/main/mydiffusion_zeggs/mydiffusion_zeggs/output/train_beat_semges_joint/vae_checkpoint_600.bin")
    semges.load_state_dict(checkpoint['state_dict'], strict=False)
    semges.eval()
    semges.cuda()



    if minibatch:
        if args.mode == 'audio_pharse':
            text_sequence = args.pharse
        elif args.mode == 'audio_text':
            text_sequence = args.locomotion_text
        print("audio shape:",audio.shape,"     num_subdivision:",num_subdivision,"     stride_poses:",stride_poses)
        audio_reshape = torch.from_numpy(audio).to(torch.float32).reshape(num_subdivision, stride_poses * int(16000 / 20)).to(mydevice).transpose(0, 1)       # mfcc[:, :-2]

        if args.mode == 'audio_only':
            shape_body = (1, 451, model.nfeats, args.n_poses)
            shape_hand = (1, 690, model.nfeats, args.n_poses)
            out_list = []
            out_list_body=[]
            out_list_hand=[]
            sss_1=torch.zeros([1,88,451])
            sss_2=torch.zeros([1,88,690])
        else:
            shape_body = (1, 238, model.nfeats, args.n_poses)
            shape_hand = (1, 756, model.nfeats, args.n_poses)
            out_list = []
            transl_out_list = []
            out_list_body=[]
            out_list_hand=[]
            sss_1=torch.zeros([1,180,238])
            sss_2=torch.zeros([1,180,756])

        for i in range(0, num_subdivision):
            print(i, num_subdivision)
            model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1]

            if i == 0:
                if n_seed != 0:
                    pad_zeros = torch.zeros([n_seed * int(16000 / 20), 1]).to(mydevice)        # wavlm dims are 1024
                    model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0)
                    if args.mode != 'audio_only':
                        model_kwargs_['y']['text']=text_sequence[i]
                        model_kwargs_['y']['speaker_id']=torch.tensor(args.speaker_id).to(mydevice).float().unsqueeze(0)
                    
                    save_audio_clip=model_kwargs_['y']['audio'].clone()
                    save_audio_clip=save_audio_clip[:,0].detach().cpu().numpy()
                    save_audio_clip=save_audio_clip.astype(np.float32)
                    sr = 16000
                    audio_path = 'test_audio.wav'
                    sf.write(audio_path, save_audio_clip, sr)
                    text = asr.file_to_text(audio_path)
                    sample_text=text[0]
                    print("sample_text:   ",sample_text)
                    model_kwargs_['y']['wav_text']=sample_text

                    #获取语义对齐特征
                    with torch.no_grad():
                        text_latent_body,text_latent_hand=semges(sample_text)
                    model_kwargs_['y']['text_latent_body']=text_latent_body
                    model_kwargs_['y']['text_latent_hand']=text_latent_hand


                    if args.mode == 'audio_only':
                        model_kwargs_['y']['seed_body'] = torch.zeros([1, 451, 1, n_seed]).to(mydevice)
                        model_kwargs_['y']['seed_hand'] = torch.zeros([1, 690, 1, n_seed]).to(mydevice)
                    else:
                        model_kwargs_['y']['seed_body'] = torch.zeros([1, 238, 1, n_seed]).to(mydevice)
                        model_kwargs_['y']['seed_hand'] = torch.zeros([1, 756, 1, n_seed]).to(mydevice)
            
            else:
                if n_seed != 0:
                    pad_audio = audio_reshape[-n_seed * int(16000 / 20):, i - 1:i]
                    model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0)
                    if args.mode != 'audio_only':
                        model_kwargs_['y']['text']=text_sequence[i]
                        model_kwargs_['y']['speaker_id']=torch.tensor(args.speaker_id).to(mydevice).float().unsqueeze(0)
                    
                    save_audio_clip=model_kwargs_['y']['audio'].clone()
                    save_audio_clip=save_audio_clip[:,0].detach().cpu().numpy()
                    save_audio_clip=save_audio_clip.astype(np.float32)
                    sr = 16000
                    audio_path = 'test_audio.wav'
                    sf.write(audio_path, save_audio_clip, sr)
                    text = asr.file_to_text(audio_path)
                    sample_text=text[0]
                    print("sample_text:   ",sample_text)
                    model_kwargs_['y']['wav_text']=sample_text

                    seed_motion=out_list[-1][..., -n_seed:].to(mydevice)
                    pose_seq=seed_motion.squeeze(2).permute(0,2,1)
                    if args.mode == 'audio_only':
                        tar_pose_body=torch.zeros([pose_seq.shape[0],pose_seq.shape[1],451])
                        tar_pose_body[:,:,:13]=pose_seq[:,:,:13]
                        tar_pose_body[:,:,13:100]=(pose_seq[:,:,13:238].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#lpos 
                        tar_pose_body[:,:,100:274]=(pose_seq[:,:,238:688].reshape(pose_seq.shape[0],pose_seq.shape[1],75,2,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#ltxy 
                        tar_pose_body[:,:,274:361]=(pose_seq[:,:,688:913].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#lvel 
                        tar_pose_body[:,:,361:448]=(pose_seq[:,:,913:1138].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#lvrt 
                        tar_pose_body[:,:,448:451]=pose_seq[:,:,1138:1141]

                        tar_pose_hand=torch.zeros([pose_seq.shape[0],pose_seq.shape[1],690])
                        tar_pose_hand[:,:,:138]=(pose_seq[:,:,13:238].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))
                        tar_pose_hand[:,:,138:414]=(pose_seq[:,:,238:688].reshape(pose_seq.shape[0],pose_seq.shape[1],75,2,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))
                        tar_pose_hand[:,:,414:552]=(pose_seq[:,:,688:913].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))
                        tar_pose_hand[:,:,552:690]=(pose_seq[:,:,913:1138].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))

                        tar_pose_body=tar_pose_body.cuda()
                        tar_pose_hand=tar_pose_hand.cuda()

                        tar_pose_body=tar_pose_body.permute(0, 2, 1).unsqueeze(2)
                        tar_pose_hand=tar_pose_hand.permute(0, 2, 1).unsqueeze(2)
                        model_kwargs_['y']['seed_body'] = tar_pose_body.to(mydevice)
                        model_kwargs_['y']['seed_hand'] = tar_pose_hand.to(mydevice)
                    else:
                        tar_pose_body=torch.zeros([pose_seq.shape[0],pose_seq.shape[1],238])
                        tar_pose_body[:,:,:13*6]=pose_seq[:,:,:13*6]
                        tar_pose_body[:,:,13*6:13*9]=pose_seq[:,:,55*6:55*6+13*3]
                        tar_pose_body[:,:,13*9:13*12]=pose_seq[:,:,55*9:55*9+13*3]
                        tar_pose_body[:,:,13*12:13*18]=pose_seq[:,:,55*12:55*12+13*6]
                        tar_pose_body[:,:,13*18:13*18+4]=pose_seq[:,:,55*18:55*18+4]

                        tar_pose_hand=torch.zeros([pose_seq.shape[0],pose_seq.shape[1],756])
                        tar_pose_hand[:,:,:42*6]=pose_seq[:,:,13*6:55*6]
                        tar_pose_hand[:,:,42*6:42*9]=pose_seq[:,:,55*6+13*3:55*9]
                        tar_pose_hand[:,:,42*9:42*12]=pose_seq[:,:,55*9+13*3:55*12]
                        tar_pose_hand[:,:,42*12:42*18]=pose_seq[:,:,55*12+13*6:55*18]

                        tar_pose_body=tar_pose_body.cuda()
                        tar_pose_hand=tar_pose_hand.cuda()

                        tar_pose_body=tar_pose_body.permute(0, 2, 1).unsqueeze(2)
                        tar_pose_hand=tar_pose_hand.permute(0, 2, 1).unsqueeze(2)
                        model_kwargs_['y']['seed_body'] = tar_pose_body.to(mydevice) *0.0
                        model_kwargs_['y']['seed_hand'] = tar_pose_hand.to(mydevice) *0.0

            model_kwargs_['y']['audio'] = wav2wavlm(wavlm_model, args, model_kwargs_['y']['audio'].transpose(0, 1), mydevice)
            sample_body,sample_hand = sample_fn(
                model,
                shape_body,
                shape_hand,
                clip_denoised=False,
                model_kwargs=model_kwargs_,
                skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
                init_image_body=None,
                init_image_hand=None,
                progress=True,
                dump_steps=None,
                noise=None,  # None, torch.randn(*shape_, device=mydevice)
                const_noise=False,
                classifier=classifier,
                use_emo_classifier=use_emo_classifier,
            )
            feats_rst_body=sample_body.squeeze(2).permute(0,2,1)
            feats_rst_hand=sample_hand.squeeze(2).permute(0,2,1)
            #[1, 1141, 1, 88]
            if args.mode == 'audio_only':
                feats_rst_full=torch.zeros([feats_rst_body.shape[0],88,1141]).cuda()

                feats_rst_full[:,:,:13]=feats_rst_body[:,:,:13]

                lpos=feats_rst_full[:,:,13:238].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],75,3)
                lpos[:,:,body_index,:]=feats_rst_body[:,:,13:100].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],29,3)
                lpos[:,:,hand_index,:]=feats_rst_hand[:,:,:138].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],46,3)
                feats_rst_full[:,:,13:238]=lpos.reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],-1)

                ltxy=feats_rst_full[:,:,238:688].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],75,2,3)
                ltxy[:,:,body_index,:]=feats_rst_body[:,:,100:274].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],29,2,3)
                ltxy[:,:,hand_index,:]=feats_rst_hand[:,:,138:414].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],46,2,3)
                feats_rst_full[:,:,238:688]=ltxy.reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],-1)

                lvel=feats_rst_full[:,:,688:913].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],75,3)
                lvel[:,:,body_index,:]=feats_rst_body[:,:,274:361].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],29,3)
                lvel[:,:,hand_index,:]=feats_rst_hand[:,:,414:552].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],46,3)
                feats_rst_full[:,:,688:913]=lvel.reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],-1)

                lvrt=feats_rst_full[:,:,913:1138].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],75,3)
                lvrt[:,:,body_index,:]=feats_rst_body[:,:,361:448].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],29,3)
                lvrt[:,:,hand_index,:]=feats_rst_hand[:,:,552:690].reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],46,3)
                feats_rst_full[:,:,913:1138]=lvrt.reshape(feats_rst_full.shape[0],feats_rst_full.shape[1],-1)

                feats_rst_full[:,:,1138:1141]=feats_rst_body[:,:,448:451]
            else:
                feats_rst_full=torch.zeros([feats_rst_body.shape[0],180,994]).cuda()

                feats_rst_full[:,:,:13*6]=feats_rst_body[:,:,:13*6]
                feats_rst_full[:,:,13*6:55*6]=feats_rst_hand[:,:,:42*6]
                feats_rst_full[:,:,55*6:55*6+13*3]=feats_rst_body[:,:,13*6:13*9]
                feats_rst_full[:,:,55*6+13*3:55*9]=feats_rst_hand[:,:,42*6:42*9]
                feats_rst_full[:,:,55*9:55*9+13*3]=feats_rst_body[:,:,13*9:13*12]
                feats_rst_full[:,:,55*9+13*3:55*12]=feats_rst_hand[:,:,42*9:42*12]
                feats_rst_full[:,:,55*12:55*12+13*6]=feats_rst_body[:,:,13*12:13*18]
                feats_rst_full[:,:,55*12+13*6:55*18]==feats_rst_hand[:,:,42*12:42*18]
                feats_rst_full[:,:,55*18:55*18+4]=feats_rst_body[:,:,13*18:13*18+4]

                transl_out_list.append(feats_rst_full[:,0,55*6:55*6+3])
                transl_out_list.append(feats_rst_full[:,-1,55*6:55*6+3])
                if i >= 1:
                    transl_cha = transl_out_list[i*2] - transl_out_list[(i-1)*2+1]
                    feats_rst_full[:,:,55*6:55*6+3] -= transl_cha.cuda()

            sample=feats_rst_full.permute(0, 2, 1).unsqueeze(2)
            if len(out_list) > 0 and n_seed != 0:
                last_poses = out_list[-1][..., -n_seed:]        # # (1, model.njoints, 1, n_seed)
                out_list[-1] = out_list[-1][..., :-n_seed]  # delete last 4 frames
                if smoothing:
                    # Extract predictions
                    if args.mode == 'audio_only':
                        last_poses_root_pos = last_poses[:, 0:3]        # (1, 3, 1, 8)
                        next_poses_root_pos = sample[:, 0:3]        # (1, 3, 1, 88)
                        root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                        predict_pos = next_poses_root_pos[..., 0]
                        delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                        sample[:, 0:3] = sample[:, 0:3] - delta_pos
                    else:
                        if text_sequence[i] == '':
                            # Extract predictions
                            last_poses_root_pos = last_poses[:, :13*6]        # (1, 3, 1, 8)
                            last_poses_root_rot = last_poses[:, 55*6:55*6+3]
                            next_poses_root_pos = sample[:, :13*6]        # (1, 3, 1, 88)
                            next_poses_root_rot = sample[:, 55*6:55*6+3]
                            root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                            predict_pos = next_poses_root_pos[..., 0]
                            delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                            sample[:, :13*6] = sample[:, :13*6] - delta_pos

                            root_pos = last_poses_root_rot[..., 0]      # (1, 3, 1)
                            predict_pos = next_poses_root_rot[..., 0]
                            delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                            sample[:, 55*6:55*6+3] = sample[:, 55*6:55*6+3] - delta_pos

                    
                        else:
                            # Extract predictions
                            last_poses_root_pos = last_poses[:, 0:6]        # (1, 3, 1, 8)
                            next_poses_root_pos = sample[:, 0:6]        # (1, 3, 1, 88)
                            root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                            predict_pos = next_poses_root_pos[..., 0]
                            delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                            sample[:, 0:6] = sample[:, 0:6] - delta_pos

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[..., j]
                    next = sample[..., j]
                    sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
            out_list.append(sample)

        if n_seed != 0:
            out_list[-1] = out_list[-1][..., :-n_seed]
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
            sampled_seq = sampled_seq[:, n_seed:]
        else:
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)

    if args.mode == 'audio_only':
        data_mean_ = np.load("./audio_only_model/mean.npz")['mean'].squeeze()
        data_std_ = np.load("./audio_only_model/std.npz")['std'].squeeze()
    elif args.mode == 'audio_pharse':
        data_mean_ = np.load("./audio_pharse_hybrid_model/mean_style.npy")
        data_std_ = np.load("./audio_pharse_hybrid_model/std_style.npy")
    elif args.mode == 'audio_text':
        data_mean_ = np.load("./audio_text_hybrid_model/mean_speakerid.npy")
        data_std_ = np.load("./audio_text_hybrid_model/std_speakerid.npy")

    data_mean = np.array(data_mean_).squeeze()
    data_std = np.array(data_std_).squeeze()
    std = np.clip(data_std, a_min=0.01, a_max=None)
    out_poses = np.multiply(sampled_seq[0], std) + data_mean
    out_poses = out_poses[:last_n_frames,:]
    if last_n_frames != out_poses.shape[0]:
        out_poses_padding = np.ones((last_n_frames-out_poses.shape[0],out_poses.shape[1])) * out_poses[-1]
        out_poses = np.concatenate((out_poses,out_poses_padding),axis = 0)
    print("out_poses shape:",out_poses.shape)

    if args.mode == 'audio_only':
        prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        if smoothing: prefix += '_smoothing'
        if SG_filter: prefix += '_SG'
        if minibatch: prefix += '_minibatch'
        prefix += '_%s' % (last_n_frames)
        prefix += '_' + str(style)
        prefix += '_' + str(seed)
        if minibatch: 
            pose2bvh(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=last_n_frames, smoothing=SG_filter, upper=args.upper)
            #pose2json(os.path.join(save_dir, prefix + '.bvh'))
            #pose2json_directly(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=last_n_frames, smoothing=SG_filter, upper=args.upper)
        else:
            pose2bvh(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=last_n_frames, smoothing=SG_filter, upper=args.upper)
            #pose2json(os.path.join(save_dir, prefix + '.bvh'))
            #pose2json_directly(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=last_n_frames, smoothing=SG_filter, upper=args.upper)
    else:
        n_poses = out_poses.shape[0]
        out_poses_lb = np.zeros((n_poses, out_poses.shape[1]))
        for i in range(out_poses_lb.shape[1]):
            out_poses_lb[:, i] = savgol_filter(out_poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal
        smplx_r6d = out_poses_lb[:,:55*6]
        smplx_r6d = smplx_r6d.reshape(-1,6)
        smplx_r6d = torch.tensor(smplx_r6d)
        smplx_rotmat = rot6d_to_rotmat(smplx_r6d)
        smplx_aa = batch_rot2aa(smplx_rotmat)
        smplx_aa = smplx_aa.cpu().numpy().reshape(-1,55,3)
        transl = out_poses_lb[:,55*6:55*6+3]
        if args.mode == 'audio_pharse':
            np.savez("our_gen_pharse.npz",poses=smplx_aa,trans=transl,betas=np.zeros(10),gender='male',mocap_framerate=30)
            pose2json2("our_gen_pharse.npz")
        elif args.mode == 'audio_text':
            np.savez("our_gen_text.npz",poses=smplx_aa,trans=transl,betas=np.zeros(10),gender='male',mocap_framerate=30)
            pose2json2("our_gen_text.npz")

def main(args, save_dir, model_path, audio_path=None, mfcc_path=None, audiowavlm_path=None, max_len=0,use_emo_classifier=0):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if audiowavlm_path != None:
        mfcc, fs = librosa.load(audiowavlm_path, sr=16000)

    elif audio_path != None and mfcc_path == None:
        # normalize_audio
        audio_name = audio_path.split('/')[-1]
        normalize_wav_path = os.path.join(save_dir, 'normalize_' + audio_name)
        cmd = ['ffmpeg-normalize', audio_path, '-o', normalize_wav_path, '-ar', '16000']
        subprocess.call(cmd)

        # MFCC, https://github.com/supasorn/synthesizing_obama_network_training
        print('extract MFCC...')
        obj = MFCC(frate=20)
        wav, fs = librosa.load(normalize_wav_path, sr=16000)
        mfcc = obj.sig2s2mfc_energy(wav, None)
        np.savez_compressed(os.path.join(save_dir, audio_name[:-4] + '.npz'), mfcc=mfcc[:, :-2])

    elif mfcc_path != None and audio_path == None:
        mfcc = np.load(mfcc_path)['mfcc']

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()

    sample_fn = diffusion.p_sample_loop     # predict x_start
    emotion = args.emotion
    style = emotion2onehot[emotion]


    wavlm_model = wavlm_init(mydevice)
    inference(args, wavlm_model, mfcc, sample_fn, model, n_frames=max_len, smoothing=True, SG_filter=True, minibatch=True, skip_timesteps=0, style=style, seed=66233,use_emo_classifier=use_emo_classifier)      # style2onehot['Happy']


if __name__ == '__main__':
    save_dir = 'sample_dir'

    parser = argparse.ArgumentParser(description='ExpGest')
    parser.add_argument('--config', default='./configs/ExpGest_config.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--model_path', type=str, default='./audio_only_model/model.pt')#/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/DiffuseStyleGesture/main/mydiffusion_zeggs/mydiffusion_zeggs/speakers_4_model/model000320000.pt
    #parser.add_argument('--model_path', type=str, default='/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/DiffuseStyleGesture/main/mydiffusion_zeggs/speakers_4_model/model000860000.pt')
    parser.add_argument('--audiowavlm_path', type=str, default='')
    parser.add_argument('--max_len', type=int, default=0)
    #parser.add_argument('--use_emo_classifier', type=int, default=0)
    #parser.add_argument('--mode', type=str, default='audio_only')
    #parser.add_argument('--emotion', type=str, default='')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))

    batch_size = 1
    if config.mode == 'audio_only':
        if config.upper:
            config.model_path = './audio_only_model_upper/model.pt' 
        if config.use_emo_classifier != 0:
            if config.emotion == '':
                print(" you need give a correct emotion (netural, happiness, anger, sadness, contempt, surprise, fear, disgust)")
                exit()
            else:
                main(config, save_dir, config.model_path, audio_path=None, mfcc_path=None, audiowavlm_path=config.audiowavlm_path, max_len=config.max_len,use_emo_classifier=config.use_emo_classifier)
        else:
            #config.model_path = '/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/DiffuseStyleGesture/main/mydiffusion_zeggs/beats_1/model000080000.pt'
            main(config, save_dir, config.model_path, audio_path=None, mfcc_path=None, audiowavlm_path=config.audiowavlm_path, max_len=config.max_len,use_emo_classifier=config.use_emo_classifier)
    elif config.mode == 'audio_pharse':
        from model.mdm_id import MDM
        config.n_poses = 180
        config.max_len = 180
        config.model_path = './audio_pharse_hybrid_model/model.pt'
        main(config, save_dir, config.model_path, audio_path=None, mfcc_path=None, audiowavlm_path=config.audiowavlm_path, max_len=config.max_len,use_emo_classifier=config.use_emo_classifier)

        
    elif config.mode == 'audio_text':
        from model.mdm_ges import MDM
        config.n_poses = 180
        config.model_path = './audio_text_hybrid_model/model.pt'
        user_text = config.locomotion_text
        config.max_len = len(user_text) * 180
        main(config, save_dir, config.model_path, audio_path=None, mfcc_path=None, audiowavlm_path=config.audiowavlm_path, max_len=config.max_len,use_emo_classifier=config.use_emo_classifier)

    else:
        print(" you need give a correct mode (audio_only, audio_pharse, audio_text) !!! ")
        exit()

