import torch
import torchvision
#from model import Generator

#UPSCALE_FACTOR = 1
#model = Generator(UPSCALE_FACTOR)
model=torch.load('160_model_path.pth', map_location=lambda storage, loc: storage)
#model.eval()
print('model:',model)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 33, 33)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("160_model.pt")
