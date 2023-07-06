import torch
import torch.nn as nn
from config.base_config import Config
from torch import nn
from modules.transformer import Transformer

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config

        config.pooling_type = 'avg'
        self.use_beat = config.use_beat
        self.mode = config.mode

        video_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim*2, nhead=config.num_mha_heads)
        music_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim*2, nhead=config.num_mha_heads)

        self.music_transformer = nn.TransformerEncoder(video_encoder_layer, num_layers=config.num_layers)
        self.video_transformer = nn.TransformerEncoder(music_encoder_layer, num_layers=config.num_layers)

        self.music_linear = nn.Linear(768, config.embed_dim)
        self.video_linear = nn.Linear(512, config.embed_dim)

        self.clip_logit_scale = torch.FloatTensor([4.6052]).cuda()

        video_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads)
        music_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads)
        self.music_beat_transformer = nn.TransformerEncoder(video_beat_encoder_layer, num_layers=config.num_layers)
        self.video_beat_transformer = nn.TransformerEncoder(music_beat_encoder_layer, num_layers=config.num_layers+2)
        self.music_beat_linear = nn.Sequential(
            nn.Linear(25, 84),
            nn.Linear(84, config.embed_dim),
        )
        self.video_beat_linear = nn.Sequential(
            nn.Linear(25, 84),
            nn.Linear(84, config.embed_dim),
        )

        self.beat_pool = Transformer(config)


    def forward(self, data):
        batch_size = data['video'].shape[0]
        music_data = data['music']
        video_data = data['video']

        music_data = (self.music_linear(music_data))
        video_data = (self.video_linear(video_data))

        music_beat = data['music_beat']
        video_beat = data['video_beat']
        music_beat = self.music_beat_linear(music_beat)
        video_beat = self.video_beat_linear(video_beat)
        video_features_trans = self.video_multimodel_fuse(video_data, video_beat)
        music_features_trans = self.music_multimodel_fuse(music_data, music_beat)

        video_beat_features = torch.mean(video_features_trans['video_beat'], dim=1)
        music_beat_features = music_features_trans['music_beat'].reshape(batch_size, self.config.num_frames, -1)
        music_beat_features_pooled = self.beat_pool(video_beat_features, music_beat_features)

        music_features_trans['music_data'] = music_features_trans['music_data'].reshape(-1, int(self.config.embed_dim*2) * (self.config.num_frames))
        music_features_trans['music_beat'] = music_features_trans['music_beat']
        music_features_trans['music_beat_pool'] = music_beat_features_pooled
        video_features_trans['video_data'] = video_features_trans['video_data'].reshape(-1, int(self.config.embed_dim*2) * (self.config.num_frames))
        video_features_trans['video_beat'] = video_features_trans['video_beat']
        video_features_trans['video_beat_pool'] = video_beat_features


        return music_features_trans, video_features_trans



    def music_multimodel_fuse(self, music_data, music_beat):

        add_data = music_data + music_beat
        mul_data = music_data * music_beat
        fuse_data = torch.cat([add_data, mul_data], dim=-1)
        music_data = self.music_transformer(fuse_data)
        music_beat = self.music_beat_transformer(music_beat)
        fuse_data = {'music_data':music_data, 'music_beat': music_beat}
        return fuse_data
        

    def video_multimodel_fuse(self, video_data=None, video_beat=None):

        add_data = video_data + video_beat
        mul_data = video_data * video_beat
        fuse_data = torch.cat([add_data, mul_data], dim=-1)               
        video_data = self.video_transformer(fuse_data)
        video_beat = self.video_beat_transformer(video_beat)
        fuse_data = {'video_data': video_data, 'video_beat': video_beat}           
        return fuse_data
