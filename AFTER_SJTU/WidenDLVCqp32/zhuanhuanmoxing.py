import torch
import torchvision
from model import WideResNet,BasicBlock,NetworkBlock


# 载入预训练模型

model = WideResNet(depth=28,widen_factor=4, dropRate=0).eval()
model.load_state_dict(torch.load('netG_epoch_1_154.pth', map_location=lambda storage, loc: storage))

print('model:',model)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 33, 33)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("netG_epoch_1_154.pt")
