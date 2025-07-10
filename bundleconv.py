import torch
import torch.nn as nn
from typing import List


class ConvBlock(nn.Module):
    """Basic convolutional block used by BundleConv (Conv → Activation → BatchNorm)."""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.bn(self.activation(self.conv(x)))


class BundleConv(nn.Module):
    """A convolutional layer composed of multiple parallel depth-wise convolution *bundles* followed by point-wise fusions.

    This layer is a drop-in replacement for the previous `NeuronBundleLayer`, providing the same behaviour while
    presenting itself as a single convolutional layer to the rest of the network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        num_bundles: int = 4,
    ) -> None:
        """Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Desired number of output channels.
        kernel_size : int, optional (default=3)
            Kernel size for the depth-wise bundle convolutions.
        num_bundles : int, optional (default=4)
            How many parallel bundles to use.
        """
        super().__init__()
        self.num_bundles = num_bundles

        # Parallel depth-wise convolutions (one per bundle)
        self.bundles: nn.ModuleList[nn.Module] = nn.ModuleList(
            [
                ConvBlock(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=in_channels,
                )
                for _ in range(num_bundles)
            ]
        )

        # Point-wise (1×1) convolutions to merge bundle outputs
        self.merge1 = ConvBlock(in_channels * num_bundles, in_channels, kernel_size=1)
        self.merge2 = ConvBlock(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Execute all bundles in parallel and concatenate along the channel dimension
        bundle_outs: List[torch.Tensor] = [bundle(x) for bundle in self.bundles]
        out = torch.cat(bundle_outs, dim=1)

        # Fuse concatenated features back to the desired channel count
        out = self.merge1(out)
        out = self.merge2(out)
        return out


__all__ = ["BundleConv"] 