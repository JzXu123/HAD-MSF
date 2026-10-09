from .vmamba import VSSM
# from vmamba import VSSM
import torch
from torch import nn
    #  input_channels=input_channels,
#     num_classes=num_classes,
#     name="test_vmunet",
#     depths=[2, 2], 
#     depths_decoder=[2, 2],
#     drop_path_rate=0.2,
#     load_ckpt_path=None 

class VMUNet(nn.Module):
    def __init__(self, shape,
                #  input_channels=3, 
                #  num_classes=1,
                 name = "xu",
                #  depths=[4,8], 
                #  depths_decoder=[8,4],
                 drop_path_rate=0.1,
                 load_ckpt_path=None,
                ):
        super().__init__()
        rows, cols, bands = shape
        self.name = name
        bands = 64
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = bands
        self.input_channels = bands
        input_channels=bands, 
        num_classes=bands,
        self.vmunet = VSSM(in_chans=64,
                           num_classes=64,
                        #    depths=depths,
                        #    depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x):
        # print('[vmunet_x]',x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        # print('[vmunet_x1]',x.shape)
        x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
        logits = self.vmunet(x)
        logits = logits.squeeze(0).permute(1, 2, 0).contiguous()  # [H, W, C]

        # print('logits',logits.shape)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
            
# input_channels =64  # 输入通道数
# num_classes = 64     # 输出类别数
# H, W = 100,100     # 输入图像的高度和宽度

# # 实例化 VMUNet 模型
# model = VMUNet(
#     input_channels=input_channels,
#     num_classes=num_classes,
#     name="test_vmunet",
#     depths=[2, 2], 
#     depths_decoder=[2, 2],
#     drop_path_rate=0.2,
#     load_ckpt_path=None  # 如果有预训练权重路径，请设置
# ).cuda()

# # 将模型设置为评估模式
# model.eval()

# # 创建一个假输入张量
# # 形状为 [B, C, H, W]，例如批量大小为1
# batch_size = 1
# dummy_input = torch.randn(batch_size, input_channels, H, W).cuda()

# # 打印输入形状
# print(f"输入张量形状: {dummy_input.shape}")

# # 前向传播
# with torch.no_grad():
#     output = model(dummy_input)

# # 打印输出形状
# print(f"输出张量形状: {output.shape}")

# # 打印输出的一些统计信息
# print(f"输出张量最大值: {output.max().item():.4f}")
# print(f"输出张量最小值: {output.min().item():.4f}")
# print(f"输出张量均值: {output.mean().item():.4f}")

