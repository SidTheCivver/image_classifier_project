from torch import nn

# State of the art image classification model

class ImageClassificationModelV2(nn.Module):
    """
    ConvNet model; architecture taken from TinyVGG model, but with dropout layers and a dropout_p that can be reset
    Args:
        input_shape: number of input channels,
        hidden_shape: number of hidden neurons,
        output_shape: number of output channels (classes),
        conv_kernel_size: size of kernels to be used by each Conv2d layer,
        conv_stride: stride of each Conv2d layer
        mp_kernel_size: size of kernels to be used by each MaxPool2d layer
        mp_stride: stride of each MaxPool2d layer
        padding: padding to be used
        dropout_p: p-value for each dropout layer
    """
    def __init__(self,
               input_shape = 3,
               hidden_shape = 40,
               output_shape = 3,
               conv_kernel_size = 3,
               conv_stride=1,
               mp_kernel_size = 2,
               mp_stride=2,
               padding=0,
               dropout_p=0.5):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_shape,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(in_channels=hidden_shape,
                    out_channels=hidden_shape,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(kernel_size=mp_kernel_size,
                        stride=mp_stride)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape,
                    out_channels=hidden_shape,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(in_channels=hidden_shape,
                    out_channels=hidden_shape,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(kernel_size=mp_kernel_size,
                        stride=mp_stride)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape * 169,
                    out_features=output_shape)
        )
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# model_5 = ImageClassificationModelV1()
# random_tensor = torch.rand((1, 3,64,64))
# model_5(random_tensor)