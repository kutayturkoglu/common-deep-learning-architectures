from torch import nn
import torch
import torch.nn.functional as F 


class Inception(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3,red5x5, out5x5, pool_proj):
        super(Inception, self).__init__()

        """ 
            Steps:
            1x1 convolution, to reduce the dimension, and pass less parameters to the subsequent(next) 
            layer fot it to learn less features. The output is a feature map with reduced dimensions(depth)

            3x3 convolution after 1x1 dimensionality reduction. As the previous one, 1x1 decreases the spatial 
            dimension, and focuses on more local variables. After 1x1, 3x3 convolution helps to focus on more 
            global variables. Output is another feature map,  capturing different aspects of the input.

            Similar to the 3x3 one, 5x5 convolution captures even more global points from the data after a 1x1 
            dimensionality reduction.

            Finally Max pooling to be able to extract the most important features from the data. 1x1 conv layer 
            refines the output of maxpool.
        """
        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size = 1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size = 1),
            nn.Conv2d(red3x3, out3x3, kernel_size = 3, padding=1)
        )


        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size = 1),
            nn.Conv2d(red5x5, out5x5, kernel_size = 5, padding=2)
        )

    
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size = 1)
        )

    def forward(self,x):
        """
            The trick of the googlenet comes from this forward. Here we can observe that we put our input x for each branch separately, and 
            then concatanate it. By this concat, we make sure that our inception module captures diverse information from multiple spatial scales, 
            so provides us a rich representation of the input
        °"""
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1,branch3x3,branch5x5,branch_pool]
        return torch.cat(outputs,1)
