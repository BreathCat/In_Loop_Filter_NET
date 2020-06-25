import torch
import torchvision

<<<<<<< HEAD:with_model_zhuanhuanmoxing.py

#UPSCALE_FACTOR = 1
#model = Generator(UPSCALE_FACTOR)
modelname = '140epoch_dense_DLVC_model1_16block'
model=torch.load('140epoch_dense_DLVC_model1_16block.pth', map_location=lambda storage, loc: storage)
#model.eval()
=======
# An instance of your model.
model = torch.load('model_path.pth', map_location=lambda storage, loc: storage)
>>>>>>> parent of 58ffad0... half shortcut in each block, and 1x1 conv in front of each block, too many parameters:zhuanhuanmoxing.py
print('model:',model)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 33, 33)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

<<<<<<< HEAD:with_model_zhuanhuanmoxing.py
traced_script_module.save(modelname+".pt")
=======
traced_script_module.save("model.pt")
>>>>>>> parent of 58ffad0... half shortcut in each block, and 1x1 conv in front of each block, too many parameters:zhuanhuanmoxing.py
