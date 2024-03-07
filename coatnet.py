# https://github.com/chinhsuanwu/coatnet-pytorch/blob/master/coatnet.py
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def conv_3x3_bn(
    inp: int, oup: int, image_size: tuple, downsample: bool = False
) -> nn.Sequential:
    """
    Creates a 3x3 convolutional layer followed by batch normalization and GELU activation.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        image_size (tuple): The size of the input image (height, width).
        downsample (bool, optional): If True, applies a stride of 2 to downsample the input. Defaults to False.

    Returns:
        nn.Sequential: A sequential model containing the convolutional layer, batch normalization, and GELU activation.
    """
    stride = 2 if downsample else 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.GELU()
    )


class PreNorm(nn.Module):
    """
    Applies normalization before performing the input function.
    """

    def __init__(self, dim: int, fn, norm: nn.Module):
        """
        Initializes the PreNorm layer.

        Args:
            dim (int): Dimension of the input to be normalized.
            fn: The function to be applied after normalization.
            norm (nn.Module): The normalization module to use (e.g., nn.LayerNorm, nn.BatchNorm2d).
        """
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    """
    Squeeze-and-Excitation (SE) block that applies channel-wise attention.
    """

    def __init__(self, inp: int, oup: int, expansion: float = 0.25):
        """
        Initializes the SE block.

        Args:
            inp (int): Number of input channels.
            oup (int): Number of output channels.
            expansion (float, optional): Expansion factor for the inner dimension. Defaults to 0.25.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    """
    A simple feed-forward network with GELU activation and dropout.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """
        Initializes the feed-forward network.

        Args:
            dim (int): Dimension of input and output.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float, optional): Dropout rate. Defaults to 0..
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block with optional downsample and squeeze-excitation.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        image_size: tuple,
        downsample: bool = False,
        expansion: int = 4,
    ):
        """
        Initializes the MBConv block.

        Args:
            inp (int): Number of input channels.
            oup (int): Number of output channels.
            image_size (tuple): The size of the input image (height, width).
            downsample (bool, optional): If True, applies downsampling. Defaults to False.
            expansion (int, optional): Expansion factor for the middle convolution layer. Defaults to 4.
        """
        super().__init__()
        self.downsample = downsample
        stride = 2 if self.downsample else 1
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            # Depthwise convolution when expansion is 1.
            self.conv = nn.Sequential(
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # Inverted bottleneck convolution with squeeze-excitation.
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        # Apply pre-normalization to the convolutional sequence.
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            # If downsampling, apply max pooling and projection in addition to the convolution.
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            # Otherwise, apply the convolution with residual connection.
            return x + self.conv(x)


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism with relative position bias.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        image_size: tuple,
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
    ):
        """
        Initializes the Attention module.

        Args:
            inp (int): Number of input channels.
            oup (int): Number of output channels.
            image_size (tuple): Size of the input image (height, width).
            heads (int, optional): Number of attention heads. Defaults to 8.
            dim_head (int, optional): Dimension of each attention head. Defaults to 32.
            dropout (float, optional): Dropout rate. Defaults to 0..
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head**-0.5

        # Parameter table for relative position bias.
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads)
        )

        # Compute relative coordinates for position bias.
        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, oup), nn.Dropout(dropout))
        else:
            self.to_out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate query, key, and value projections.
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Compute dot product attention with scaling.
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply relative position bias.
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = rearrange(
            relative_bias,
            "(h w) c -> 1 c h w",
            h=self.ih * self.iw,
            w=self.ih * self.iw,
        )
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """
    Transformer block combining multi-head self-attention and a feed-forward network.
    """

    def __init__(
        self,
        inp: int,
        oup: int,
        image_size: tuple,
        heads: int = 8,
        dim_head: int = 32,
        downsample: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initializes the Transformer block.

        Args:
            inp (int): Number of input channels.
            oup (int): Number of output channels.
            image_size (tuple): Size of the input image (height, width).
            heads (int, optional): Number of attention heads. Defaults to 8.
            dim_head (int, optional): Dimension of each attention head. Defaults to 32.
            downsample (bool, optional): If True, applies downsampling. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to 0..
        """
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        # Apply pre-normalization and rearrangement for attention and feed-forward operations.
        self.attn = nn.Sequential(
            Rearrange("b c ih iw -> b (ih iw) c"),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw),
        )

        self.ff = nn.Sequential(
            Rearrange("b c ih iw -> b (ih iw) c"),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply downsampling, attention, and feed-forward operations accordingly.
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    """
    CoAtNet model combining convolutional and transformer blocks for image classification.
    """

    def __init__(
        self,
        image_size: tuple,
        in_channels: int,
        num_blocks: list,
        channels: list,
        num_classes: int = 1000,
        block_types: list = ["C", "C", "T", "T"],
    ):
        """
        Initializes the CoAtNet model.

        Args:
            image_size (tuple): Size of the input image (height, width).
            in_channels (int): Number of input channels.
            num_blocks (list): Number of blocks in each stage.
            channels (list): Number of channels in each stage.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            block_types (list, optional): Types of blocks in each stage ('C' for convolutional, 'T' for transformer). Defaults to ['C', 'C', 'T', 'T'].
        """
        super().__init__()
        ih, iw = image_size
        block = {"C": MBConv, "T": Transformer}

        # Construct each stage of the model with the specified number of blocks and channels.
        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2)
        )
        self.s1 = self._make_layer(
            block[block_types[0]],
            channels[0],
            channels[1],
            num_blocks[1],
            (ih // 4, iw // 4),
        )
        self.s2 = self._make_layer(
            block[block_types[1]],
            channels[1],
            channels[2],
            num_blocks[2],
            (ih // 8, iw // 8),
        )
        self.s3 = self._make_layer(
            block[block_types[2]],
            channels[2],
            channels[3],
            num_blocks[3],
            (ih // 16, iw // 16),
        )
        self.s4 = self._make_layer(
            block[block_types[3]],
            channels[3],
            channels[4],
            num_blocks[4],
            (ih // 32, iw // 32),
        )

        self.pool = nn.AvgPool2d(
            ih // 32, 1
        )  # Global average pooling before the classifier.
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)  # Classifier layer.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through each stage of the model.
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        # Apply global average pooling and classifier.
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(
        self, block, inp: int, oup: int, depth: int, image_size: tuple
    ) -> nn.Sequential:
        """
        Creates a sequence of blocks for a stage in the model.

        Args:
            block: The block type to use (convolutional or transformer).
            inp (int): Number of input channels for the first block.
            oup (int): Number of output channels for each block.
            depth (int): Number of blocks in the sequence.
            image_size (tuple): Size of the input image (height, width) for this stage.

        Returns:
            nn.Sequential: A sequential container of the specified blocks.
        """
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


# Various configurations of the CoAtNet model.
def coatnet_0() -> CoAtNet:
    """
    Constructs CoAtNet model with configuration 0.

    Returns:
        CoAtNet: CoAtNet model instance.
    """
    num_blocks = [2, 2, 3, 5, 2]
    channels = [64, 96, 192, 384, 768]
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_1() -> CoAtNet:
    """
    Constructs CoAtNet model with configuration 1.

    Returns:
        CoAtNet: CoAtNet model instance.
    """
    num_blocks = [2, 2, 6, 14, 2]
    channels = [64, 96, 192, 384, 768]
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_2() -> CoAtNet:
    """
    Constructs CoAtNet model with configuration 2.

    Returns:
        CoAtNet: CoAtNet model instance.
    """
    num_blocks = [2, 2, 6, 14, 2]
    channels = [128, 128, 256, 512, 1026]
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_3() -> CoAtNet:
    """
    Constructs CoAtNet model with configuration 3.

    Returns:
        CoAtNet: CoAtNet model instance.
    """
    num_blocks = [2, 2, 6, 14, 2]
    channels = [192, 192, 384, 768, 1536]
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def coatnet_4() -> CoAtNet:
    """
    Constructs CoAtNet model with configuration 4.

    Returns:
        CoAtNet: CoAtNet model instance.
    """
    num_blocks = [2, 2, 12, 28, 2]
    channels = [192, 192, 384, 768, 1536]
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (nn.Module): The model to count parameters for.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
