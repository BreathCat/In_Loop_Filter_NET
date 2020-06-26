import torch
import torchvision


#UPSCALE_FACTOR = 1
#model = Generator(UPSCALE_FACTOR)
modelname = '140epoch_dense_DLVC_model1_16block'
model=torch.load('140epoch_dense_DLVC_model1_16block.pth', map_location=lambda storage, loc: storage)
#model.eval()
print('model:',model)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 33, 33)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save(modelname+".pt")
