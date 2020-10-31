import torch
import torchvision
import argparse
#from model import Generator
from Wide_Res.model import BasicBlock,NetworkBlock,WideResNet

#UPSCALE_FACTOR = 1
#model = Generator(UPSCALE_FACTOR)
parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, default=64,help="model name without '.pth")
args = parser.parse_args()

if 'dict' in args.name:
    if 'Wide' in args.name:
        model = WideResNet(28, widen_factor = 4, dropRate=0).eval()
        model.load_state_dict(torch.load(args.name+'.pth', map_location=lambda storage, loc: storage))
        #model.eval()
        print('dict_model:',model)

        # An example input you would normally provide to your model's forward() method.
        example = torch.rand(1, 1, 33, 33)

        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        traced_script_module = torch.jit.trace(model, example)

        traced_script_module.save(args.name+".pt")

else:
    # modelname = 'DLVC_epoch160'
    model=torch.load(args.name+'.pth', map_location=lambda storage, loc: storage)
    #model.eval()
    print('model:',model)

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 1, 33, 33)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save(args.name+".pt")