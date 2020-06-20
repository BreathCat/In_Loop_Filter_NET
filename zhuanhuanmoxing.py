import torch
import torchvision

# An instance of your model.
model = torch.load('model_path.pth', map_location=lambda storage, loc: storage)
print('model:',model)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 33, 33)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("model.pt")
