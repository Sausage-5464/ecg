import torch    
import torch.nn as nn
import torch.nn.functional as F
from utils import FFT_for_Period, Wavelet_for_Scales
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, config):    ##configs is the configuration defined for TimesBlock
        super(TimesBlock, self).__init__() 
        self.seq_len = config['seg_len']   ##sequence length 
        
        # self.pred_len = timesblockconfig.pred_len ##prediction length
        
        self.k = config['top_k']    ##k denotes how many top frequencies are taken into consideration
        
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(config['d_model'], config['d_ff'],num_kernels=config['num_kernels']),
            nn.GELU(),
            Inception_Block_V1(config['d_ff'],config['d_model'],num_kernels=config['num_kernels'])
        )
    
    def forward(self, x):
        B, T, N = x.shape#B: batch size  T: length of time series  N:number of features
        period_list, period_weight = FFT_for_Period(x, self.k) 
        # period_list([top_k]) denotes the top_k-significant period and period_weight([B, top_k]) denotes its weight(amplitude)
        
        res = []
        for i in range(self.k):
            period = period_list[i]

            # padding : to form a 2D map, we need total length of the sequence to be divisible by the period, so padding is needed
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x

            # reshape: we need each channel of a single piece of data to be a 2D variable,
            # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions 
            # to be convolutioned to the last 2 dimensions, by calling the permute() func.
            # Whereafter, to make the tensor contiguous in memory, call contiguous()
            out = out.reshape(B, length // period, period,N).permute(0, 3, 1, 2).contiguous()
            #2D convolution to grap the intra- and inter- period information
            out = self.conv(out)

            # reshape back, similar to reshape
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            #truncating down the padded part of the output and put it to result
            
            # res.append(out[:, :(self.seq_len + self.pred_len), :])
            res.append(out[:, :self.seq_len, :])
            
        res = torch.stack(res, dim=-1) #res: 4D [B, length , 1, top_k]
        

        # adaptive aggregation
        #First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        period_weight = F.softmax(period_weight, dim=1) 

        #after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        
        #add by weight the top_k periods' result, getting the result of this TimesBlock
        res = torch.sum(res * period_weight, -1)

        # residual connection
        
        res = res + x
        return res
 
class TimesModel(nn.Module):
    def __init__(self, config):
        super(TimesModel, self).__init__()
        
        self.projection = nn.Linear(config['d_model'] * config['seg_len'], len(config['categories']))
        # 使用 nn.Sequential 来存储多个 TimesBlock
        self.timesblocks = nn.Sequential(*[TimesBlock(config) for _ in range(config['n_layers'])])
        
        self.enc_embedding = DataEmbedding(1, config['d_model'],config['drop_out'])
        

    def forward(self, x):
        x = self.enc_embedding(x)  # [b,t,1] -> [B,T,C]
        
        # 通过所有的 TimesBlock
        x = self.timesblocks(x) # [B,T,C]
        
        x = x.view(x.size(0), -1) # [B,T,C] -> [B, T*C]
        
        x = self.projection(x) # [B, T*C] -> [B, n_classes]
        
        return x

class WaveletBlock(nn.Module):
    def __init__(self, config):
        super(WaveletBlock, self).__init__()
        self.seq_len = config['seg_len']
        self.top_k = config['top_k']
        self.wavelet_type = config['wavelet']
        
        # 2D处理模块 (保持与TimesBlock相同结构)
        self.conv = nn.Sequential(
            Inception_Block_V1(config['d_model'], config['d_ff'], 
                             num_kernels=config['num_kernels']),
            nn.GELU(),
            Inception_Block_V1(config['d_ff'], config['d_model'], 
                             num_kernels=config['num_kernels'])
        )
    
    def forward(self, x):
        B, T, N = x.shape
        scale_list, scale_weight = Wavelet_for_Scales(x, self.top_k,self.wavelet_type)
        
        res = []
        for i in range(self.top_k):
            scale = scale_list[i]
            
            # 填充使长度可被scale整除
            if T % scale != 0:
                length = ((T // scale) + 1) * scale
                padding = torch.zeros([B, length-T, N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            
            # 重塑为2D: [B, length//scale, scale, N] -> [B, N, length//scale, scale]
            out = out.reshape(B, length//scale, scale, N).permute(0,3,1,2).contiguous()
            
            # 2D卷积处理
            out = self.conv(out)
            
            # 恢复为1D
            out = out.permute(0,2,3,1).reshape(B, -1, N)
            res.append(out[:, :T, :])  # 截断填充部分
        
        # 加权融合
        res = torch.stack(res, dim=-1)
        scale_weight = F.softmax(scale_weight, dim=1).unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * scale_weight, dim=-1)
        
        # 残差连接
        res = res + x
        return res

class WaveletModel(nn.Module):
    def __init__(self, config):
        super(WaveletModel, self).__init__()
        self.config = config
        
        # 数据嵌入层
        self.enc_embedding = DataEmbedding(1, config['d_model'], config['drop_out'])
        
        # 堆叠WaveletBlocks
        self.wavelet_blocks = nn.Sequential(*[
            WaveletBlock(config) for _ in range(config['n_layers'])
        ])
        
        # 分类头
        self.projection = nn.Linear(config['d_model'] * config['seg_len'], 
                                  len(config['categories']))
    
    def forward(self, x):
        x = self.enc_embedding(x)  # [B,T,1] -> [B,T,d_model]
        x = self.wavelet_blocks(x)  # 多尺度小波处理
        x = x.reshape(x.size(0), -1)  # [B,T*d_model]
        x = self.projection(x)  # 分类预测
        return x
