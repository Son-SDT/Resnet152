
import sys

sys.path.append("...")

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn


def cal_padding(tensor_size,K,S):
    return round((S*(tensor_size -1) -tensor_size+K )/2)

class identity_block1(nn.Module):
    def __init__(self,input,filter,tensor_size ):
        super().__init__()
        F1,F2,F3 = filter

        self.Block1 = nn.Sequential(
            # Frist layer
            nn.Conv2d(input,F1, kernel_size = 1 , stride =1, padding = cal_padding(tensor_size,1,1)),
            nn.BatchNorm2d(F1),
            nn.ReLU(),

            # Second layer
            nn.Conv2d(F1,F2, kernel_size = 3, stride =1, padding = cal_padding(tensor_size,3,1)),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            
            # Third layer
            nn.Conv2d(F2,F3, kernel_size = 1 , stride =1, padding = cal_padding(tensor_size,1,1)),
            nn.BatchNorm2d(F3),
        )
        self.X_shortcut = nn.Sequential(
            nn.Conv2d(input,F3,1,1),
            nn.BatchNorm2d(F3)
        )
        self.relu = nn.ReLU()    
        
    def forward(self,x):
        X_shortcut = self.X_shortcut(x)
        x = self.Block1(x)
        x = torch.add(x,X_shortcut)
        x = self.relu(x)
        
        return x

class identity_block2(nn.Module):
    def __init__(self,input,filter,tensor_size ):
        super().__init__()
        F1,F2,F3 = filter
        self.Block2 = nn.Sequential(
        # Frist layer
        nn.Conv2d(input,F1, kernel_size = 1 , stride =1, padding = cal_padding(tensor_size,1,1)),
        nn.BatchNorm2d(F1),
        nn.ReLU(),

        # Second layer
        nn.Conv2d(F1,F2, kernel_size = 3, stride =1, padding = cal_padding(tensor_size,3,1)),
        nn.BatchNorm2d(F2),
        nn.ReLU(),
        
        # Third layer
        nn.Conv2d(F2,F3, kernel_size = 1 , stride =1, padding = cal_padding(tensor_size,1,1)),
        nn.BatchNorm2d(F3),
        )
        self.relu = nn.ReLU()

    def forward(self,x):
        X_shortcut = x
        x = self.Block2(x)
        x = torch.add(x,X_shortcut)
        x = self.relu(x)
        
        return x

class Resnet152 (nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        stage1 = [64,64,256]
        stage2 = [128,128,512]
        stage3 = [256,256,1024]
        stage4 = [512,512,2048]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channel,64, kernel_size = 7, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage_1_b1 = identity_block1(64,stage1,56)
        self.stage_1_b2 = identity_block2(256,stage1,56)
        self.stage_1_b3 = identity_block2(256,stage1,56)
        
        self.stage_2_b1 = identity_block1(256,stage2,28)
        self.stage_2_b2 = identity_block2(512,stage2,28)
        self.stage_2_b3 = identity_block2(512,stage2,28)
        self.stage_2_b4 = identity_block2(512,stage2,28)
        self.stage_2_b5 = identity_block2(512,stage2,28)
        self.stage_2_b6 = identity_block2(512,stage2,28)
        self.stage_2_b7 = identity_block2(512,stage2,28)
        self.stage_2_b8 = identity_block2(512,stage2,28)

        
        self.stage_3_b1 = identity_block1(512,stage3,14)
        self.stage_3_b2 = identity_block2(1024,stage3,14)
        self.stage_3_b3 = identity_block2(1024,stage3,14)
        self.stage_3_b4 = identity_block2(1024,stage3,14)
        self.stage_3_b5 = identity_block2(1024,stage3,14)
        self.stage_3_b6 = identity_block2(1024,stage3,14)
        self.stage_3_b7 = identity_block2(1024,stage3,14)
        self.stage_3_b8 = identity_block2(1024,stage3,14)
        self.stage_3_b9 = identity_block2(1024,stage3,14)
        self.stage_3_b10 = identity_block2(1024,stage3,14)
        self.stage_3_b11 = identity_block2(1024,stage3,14)
        self.stage_3_b12 = identity_block2(1024,stage3,14)
        self.stage_3_b13 = identity_block2(1024,stage3,14)
        self.stage_3_b14 = identity_block2(1024,stage3,14)
        self.stage_3_b15 = identity_block2(1024,stage3,14)
        self.stage_3_b16 = identity_block2(1024,stage3,14)
        self.stage_3_b17 = identity_block2(1024,stage3,14)
        self.stage_3_b18 = identity_block2(1024,stage3,14)
        self.stage_3_b19 = identity_block2(1024,stage3,14)
        self.stage_3_b20 = identity_block2(1024,stage3,14)
        self.stage_3_b21 = identity_block2(1024,stage3,14)
        self.stage_3_b22 = identity_block2(1024,stage3,14)
        self.stage_3_b23 = identity_block2(1024,stage3,14)
        self.stage_3_b24 = identity_block2(1024,stage3,14)
        self.stage_3_b25 = identity_block2(1024,stage3,14)
        self.stage_3_b26 = identity_block2(1024,stage3,14)
        self.stage_3_b27 = identity_block2(1024,stage3,14)
        self.stage_3_b28 = identity_block2(1024,stage3,14)
        self.stage_3_b29 = identity_block2(1024,stage3,14)
        self.stage_3_b30 = identity_block2(1024,stage3,14)
        self.stage_3_b31 = identity_block2(1024,stage3,14)
        self.stage_3_b32 = identity_block2(1024,stage3,14)
        self.stage_3_b33 = identity_block2(1024,stage3,14)
        self.stage_3_b34 = identity_block2(1024,stage3,14)
        self.stage_3_b35 = identity_block2(1024,stage3,14)
        self.stage_3_b36 = identity_block2(1024,stage3,14)

        self.stage_4_b1 = identity_block1(1024,stage4,7)
        self.stage_4_b2 = identity_block2(2048,stage4,7)
        self.stage_4_b3 = identity_block2(2048,stage4,7)
        
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride = 2),
            
        )    
        self.linear = nn.Linear(2048,num_classes)
        self.max_pool2D = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.stem(x)
        #print(x.shape)
        x = self.max_pool2D(x)
        #print("Before stage",x.shape)

        x = self.stage_1_b1(x)
        x = self.stage_1_b2(x)
        x = self.stage_1_b3(x)
        x = self.max_pool2D(x)
        #print("Stage 1",x.shape)

        x = self.stage_2_b1(x)
        x = self.stage_2_b2(x)
        x = self.stage_2_b3(x)
        x = self.stage_2_b4(x)
        x = self.stage_2_b5(x)
        x = self.stage_2_b6(x)
        x = self.stage_2_b7(x)
        x = self.stage_2_b8(x)
        x = self.max_pool2D(x)
        #print("Stage 2",x.shape)

        x = self.stage_3_b1(x)
        x = self.stage_3_b2(x)
        x = self.stage_3_b3(x)
        x = self.stage_3_b4(x)
        x = self.stage_3_b5(x)
        x = self.stage_3_b6(x)
        x = self.stage_3_b7(x)
        x = self.stage_3_b8(x)
        x = self.stage_3_b9(x)
        x = self.stage_3_b10(x)
        x = self.stage_3_b11(x)
        x = self.stage_3_b12(x)
        x = self.stage_3_b13(x)
        x = self.stage_3_b14(x)
        x = self.stage_3_b15(x)
        x = self.stage_3_b16(x)
        x = self.stage_3_b17(x)
        x = self.stage_3_b18(x)
        x = self.stage_3_b19(x)
        x = self.stage_3_b20(x)
        x = self.stage_3_b21(x)
        x = self.stage_3_b22(x)
        x = self.stage_3_b23(x)
        x = self.stage_3_b24(x)
        x = self.stage_3_b25(x)
        x = self.stage_3_b26(x)
        x = self.stage_3_b27(x)
        x = self.stage_3_b28(x)
        x = self.stage_3_b29(x)
        x = self.stage_3_b30(x)
        x = self.stage_3_b31(x)
        x = self.stage_3_b32(x)
        x = self.stage_3_b33(x)
        x = self.stage_3_b34(x)
        x = self.stage_3_b35(x)
        x = self.stage_3_b36(x)
        x = self.max_pool2D(x)
        #print("Stage 3",x.shape)

        x = self.stage_4_b1(x)
        x = self.stage_4_b2(x)
        x = self.stage_4_b3(x)
        #print("Stage 4",x.size())
        x = self.classifier(x)
        #print("Avg pool",x.shape)
        #x = x.view(x.shape[0],-1)
        x = self.flatten(x)
        #print("Flatten size",x.shape)
        x = self.linear(x)
        #print("Output",x.shape)
        return x

test_net = Resnet152(3,10)
test_x = Variable(torch.zeros(8, 3, 229, 229))

x = test_net(test_x)
print(x.shape)
