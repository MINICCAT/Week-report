# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,  # 输入图像的大小
        patch_size: int = 16,  # 将图像分割成的小块大小
        in_chans: int = 3,  # 输入图像的通道数
        embed_dim: int = 768,  # 嵌入向量的维度，即Transformer的输入维度，通常为768
        depth: int = 12,  # Transformer编码器的深度，要用几个transformer block，类似于重复几次
        num_heads: int = 12,  # 在每个Vit block中多头注意力机制中的头数
        mlp_ratio: float = 4.0,  # MLP隐藏层维度与嵌入维度的比例
        out_chans: int = 256,  # 输出通道数
        qkv_bias: bool = True,  # 是否在查询、键、值投影中添加偏置，如果是，则会在查询、键、值投影中添加可学习的偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,  # 标准化层类型
        act_layer: Type[nn.Module] = nn.GELU,  # 激活函数类型
        use_abs_pos: bool = True,  # 是否使用绝对位置编码
        use_rel_pos: bool = False,  # 是否在注意力图中添加相对位置编码
        rel_pos_zero_init: bool = True,  # 是否零初始化相对位置参数
        window_size: int = 0,  # 窗口注意力块的窗口大小
        global_attn_indexes: Tuple[int, ...] = (),  # 使用全局注意力的块索引
    ) -> None:
        
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size), #解释：将图像分割成patch时，卷积核的大小
            stride=(patch_size, patch_size), #解释：将图像分割成patch时，每次移动的步长
            in_chans=in_chans, #解释：输入图像的通道数
            embed_dim=embed_dim, #解释：嵌入向量的维度，即输出的维度
        )

        self.pos_embed: Optional[nn.Parameter] = None #解释：绝对位置编码
        if use_abs_pos: # 初始化绝对位置编码，默认都为0，并且他的形状是使用预训练图像大小
            self.pos_embed = nn.Parameter(  torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)   )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, #解释：Transformer编码器的输入维度
                num_heads=num_heads,  #解释：Transformer编码器中多头注意力机制的头数
                mlp_ratio=mlp_ratio,  #解释：Transformer编码器中MLP隐藏层维度与嵌入维度的比例
                qkv_bias=qkv_bias,  #解释：是否在查询、键、值投影中添加偏置
                norm_layer=norm_layer, #解释：标准化层类型
                act_layer=act_layer,  #解释：激活函数类型
                use_rel_pos=use_rel_pos,  #解释：是否在注意力图中添加相对位置编码
                rel_pos_zero_init=rel_pos_zero_init,  #解释：是否零初始化相对位置参数
                window_size=window_size if i not in global_attn_indexes else 0, #解释：窗口注意力块的窗口大小,如果当前块索引在全局注意力索引中，则窗口大小为0
                input_size=(img_size // patch_size, img_size // patch_size), #解释：输入图像的大小
            )
            self.blocks.append(block)

        self.neck = nn.Sequential( #解释：Transformer编码器的输出层，将Transformer编码器的输出进行卷积和标准化
            nn.Conv2d(  embed_dim, out_chans, kernel_size=1, bias=False,  ),
            LayerNorm2d(out_chans),
            nn.Conv2d(  out_chans, out_chans, kernel_size=3, padding=1, bias=False, ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) #用patch_embed函数将输入图像分割成patch，并嵌入到向量空间中
        if self.pos_embed is not None:
            x = x + self.pos_embed #将绝对位置编码的信息新增到x中

        for blk in self.blocks:
            x = blk(x) #将x送入到每一个block中进行处理

        x = self.neck(x.permute(0, 3, 1, 2)) #将x的维度进行转换，并送入到neck中进行处理

        return x

### 重要！！ 下面的所有函数都是ViT块中用到的辅助函数

class Block(nn.Module): # 这个block就是手搓Transformer块，支持窗口注意力和残差传播块

    def __init__(
        self, #初始化函数，用于创建ViT块的各个组件
        dim: int,  # 输入通道的数量
        num_heads: int,  # 每个ViT块中注意力头的数量
        mlp_ratio: float = 4.0,  # MLP隐藏层维度与嵌入维度的比例
        qkv_bias: bool = True,  # 是否在查询、键、值中添加可学习的偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,  # 归一化层
        act_layer: Type[nn.Module] = nn.GELU,  # 激活层
        use_rel_pos: bool = False,  # 是否在注意力图中添加相对位置嵌入
        rel_pos_zero_init: bool = True,  # 是否将相对位置参数初始化为零
        window_size: int = 0,  # 窗口注意力块的窗口大小，如果为0则使用全局注意力
        input_size: Optional[Tuple[int, int]] = None,  # 输入分辨率，用于计算相对位置参数大小，只有在使用相对位置时才启用
    ) -> None:
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

# 这个attention是用在block中的，支持窗口注意力和相对位置编码
class Attention(nn.Module): #这个类是多头注意力机制的模块，并且带有相对位置编码
    def __init__(
        self,
        dim: int, # 维度，从前面的代码来看，这里的维度为768
        num_heads: int = 8, # 每个ViT块中注意力头的数量，注意力头的数量必须能整除输入维度dim
        qkv_bias: bool = True,  # 是否在查询、键、值中添加可学习的偏置，如果为True，则会在查询、键、值中添加可学习的偏置
        use_rel_pos: bool = False, # 是否在注意力图中添加相对位置嵌入，如果为True，则会在注意力图中添加相对位置嵌入
        rel_pos_zero_init: bool = True, # 是否将相对位置参数初始化为零，如果为True，则将相对位置参数初始化为零
        input_size: Optional[Tuple[int, int]] = None, # 输入分辨率，用于计算相对位置参数大小
    ) -> None:

        super().__init__() #调用父类 nn.Module的构造函数，这是必须的步骤
        self.num_heads = num_heads # 8
        head_dim = dim // num_heads # 768 // 8 = 96
        self.scale = head_dim**-0.5 # 1 / sqrt(96) = 0.099503617

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #这是一个线性层（全连接层），它会一次性生成查询（Query）、键（Key）和值（Value）三个矩阵。因为是dim * 3，所以它的输出大小是原始维度的三倍
        self.proj = nn.Linear(dim, dim) #这是最终的线性投影层，用于将多头注意力的输出整合回原始维度。

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # 如果 use_rel_pos为 True，则会初始化两个参数 self.rel_pos_h和 self.rel_pos_w，分别对应高度和方向上的相对位置嵌入。它们被定义为 nn.Parameter，这意味着这些张量在模型训练过程中会被优化
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # 初始化相对位置编码
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor: #输入是张量 x，输出也是张量
        B, H, W, _ = x.shape #获取输入张量 x的维度，分别代表批大小（B）、高度（H）、宽度（W）。_表示最后一个维度（特征维度），我们不需要显式使用它的值。
        
        # 这里是核心步骤，需要谨记
        # qkv with shape (3, B, nHead, H * W, C)
        # self.qkv(x)通过线性层得到Q、K、V的合并结果。
        # .reshape(...)将输出重塑，为分割多头做准备。
        # .permute(...)调整维度的顺序，使得Q、K、V分离，并便于后续计算
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        
        # q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)：最后将张量重新变形并拆分成查询（q）、键（k）、值（v）三个独立的张量。
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # 这里就用上了数学公式，q乘缩放因子 ，再与K的转置相乘
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

## 图像编码中用到的第一个函数
class PatchEmbed(nn.Module): 
    """
    将图像分割成小块并转换为固定维度向量
    投影层（projection layer）通常指的是用于将高维数据映射到低维空间或进行特征变换的层。在卷积神经网络（CNN）中，投影层通常指的是卷积层（Convolutional Layer）或全连接层（Fully Connected Layer）。这些层可以学习到输入数据的局部特征，并将它们映射到新的特征空间中。
    """
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),  # 投影层的卷积核大小，16x16，决定最后输出的图像块的大小为16*16
        stride: Tuple[int, int] = (16, 16), # 步长（stride）决定了块之间的重叠程度，这里是16*16，stride等于kernel_size说明没有重叠
        padding: Tuple[int, int] = (0, 0), #投影层的填充大小，默认不填充
        in_chans: int = 3, # 输入通道数，默认为3（RGB图像）
        embed_dim: int = 768, # 输出通道数，这里是作者设定的默认图像最后输出的通道为768，embed_dim的选择会影响后续Transformer模型的计算复杂度
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(  in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C 转换输出内容的顺序
        x = x.permute(0, 2, 3, 1)
        return x
