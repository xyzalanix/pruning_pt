#https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import pickle
import dnnlib
import array
import legacy
from torch_utils import misc
import random
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Gs_kwargs = dnnlib.EasyDict()

with dnnlib.util.open_url("yugioh.pkl") as f:
    # G = legacy.load_network_pkl(f, **Gs_kwargs)['G_ema'].to(device) # this works but can't save only generator without keys()?
    model = legacy.load_network_pkl(f) # 

# print("Dirs: ",G.__dir__())
module = model['G_ema']
# print("Keys: ",G.keys())
print("Named Modules: ", list(module.named_modules()))
# module = module.synthesis
# print("Type: ", type(module))
# print("Named Parameters: ", list(module.named_parameters()))
# print(module.__dir__()) 

# TODO: Pick a random layer. Not working cause takes string literally and not as a param.

# module_list = list(module.named_modules())
# n_prune = 1
# print('Pruning', n_prune,'out of', len(module_list), 'modules at random.')
# # random_pick = random.choices(module_list, k=n_prune)
# random_pick = module_list[44]
# # print(type(random_pick))
# # random_pick = random_pick.pop(0)
# # a, b = random_pick
# random_pick = random_pick[0]
# print(random_pick)
# # print(a)
# # module = module.a
# print(module)

# Method 1 : Local Unstructured Pruning
# prune.random_unstructured(module.synthesis.L9_532_128, name="weight", amount=0.5)
# print("-.-.-.-.-. ", list(module.named_parameters()))
# print("-.-.-.-.-. ", list(module.named_buffers()))
# print("-.-.-.-.-. ", module.weight)
# print("-.-.-.-.-. ", module._forward_pre_hooks)
#prune.l1_unstructured(module, name="bias", amount=5)
# print("-.-.-.-.-. ", list(module.named_parameters()))
# print("-.-.-.-.-. ", list(module.named_buffers()))
# print("-.-.-.-.-. ", module.bias)
# print("-.-.-.-.-. ", module._forward_pre_hooks)

## Now we remove originals and make pruning permanent
# prune.remove(module, 'weight')
# prune.remove(module, 'bias')
# print(G.state_dict().keys())

# Method 2 : Global Unstructured Pruning

# parameters = (
#     (module.mapping.fc1, "weight"),
#     (module.mapping.fc1, "bias"),
#     (module.synthesis.b64.conv0, "weight"),
#     (module.synthesis.b64.conv0, "bias"),
#     (module.synthesis.b64.conv1, "weight"),
#     (module.synthesis.b64.conv1, "bias"),
#     (module.mapping.fc4, "weight"),
#     (module.mapping.fc4, "bias"),
# )
# prune.global_unstructured(
#     parameters,
#     pruning_method=prune.L1Unstructured,
#     amount=1,
# )

# prune.remove(module.mapping.fc1, "weight")
# prune.remove(module.mapping.fc1, "bias")
# prune.remove(module.synthesis.b64.conv0, "weight")
# prune.remove(module.synthesis.b64.conv0, "bias")
# prune.remove(module.synthesis.b64.conv1, "weight")
# prune.remove(module.synthesis.b64.conv1, "bias")
# prune.remove(module.mapping.fc4, "weight")
# prune.remove(module.mapping.fc4, "bias")

# Method 3 : Local Structured Pruning
# prune.ln_structured(module.mapping.fc0, "weight", amount=0.1, n=2, dim=1)
# prune.remove(module.mapping.fc0, "weight")
# prune.ln_structured(module.mapping.fc1, "weight", amount=0.1, n=2, dim=1)
# prune.remove(module.mapping.fc1, "weight")

# prune.ln_structured(module.mapping.fc7, "weight", amount=0.1, n=2, dim=1)
# prune.remove(module.mapping.fc7, "weight")
# prune.ln_structured(module.mapping.fc6, "weight", amount=0.1, n=2, dim=1)
# prune.remove(module.mapping.fc6, "weight")

# prune.ln_structured(module.mapping.fc0, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc0, "weight")
# prune.ln_structured(module.mapping.fc1, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc1, "weight")
# prune.ln_structured(module.mapping.fc7, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc7, "weight")
# prune.ln_structured(module.mapping.fc6, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc6, "weight")
# prune.ln_structured(module.mapping.fc2, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc2, "weight")
# prune.ln_structured(module.mapping.fc3, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc3, "weight")
# prune.ln_structured(module.mapping.fc4, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc4, "weight")
# prune.ln_structured(module.mapping.fc5, "weight", amount=100, n=2, dim=0)
# prune.remove(module.mapping.fc5, "weight")

# prune.ln_structured(module.synthesis.b64.conv0, "weight", amount=100, n=2, dim=1)
# prune.remove(module.synthesis.b64.conv0, "weight")
# prune.ln_structured(module.synthesis.b64.conv1, "weight", amount=100, n=2, dim=1)
# prune.remove(module.synthesis.b64.conv1, "weight")


# # SG3
# # pruning lower layers impacta la forma, resulta en blobs.

# # Randomize amount values
amount0 = random.randint(1,128)
amount1 = random.randint(1,81)
amount2 = random.randint(1,51)
amount3 = random.randint(1,32)
amount4 = random.randint(1,512)
amount5 = random.randint(1,512)
amount6 = random.randint(1,51)

# what's the # of parameters to prune ?
pruning_strength=(str(amount1), str(amount2), str(amount3), str(amount4), str(amount5), str(amount6))
pruning_strength = " ".join(str(x) for x in pruning_strength)
pruning_strength=pruning_strength.replace(" ","_")
print(pruning_strength)

## Yugioh Layers
# prune.ln_structured(module.synthesis.L3_52_512, "weight", amount=amount4, n=2, dim=1)
# prune.remove(module.synthesis.L3_52_512, "weight")
# prune.ln_structured(module.synthesis.L5_148_512, "weight", amount=amount5, n=2, dim=1)
# prune.remove(module.synthesis.L5_148_512, "weight")
prune.ln_structured(module.synthesis.L9_532_128, "weight", amount=amount0, n=2, dim=1)
prune.remove(module.synthesis.L9_532_128, "weight")
prune.ln_structured(module.synthesis.L10_1044_81, "weight", amount=amount1, n=2, dim=1)
prune.remove(module.synthesis.L10_1044_81, "weight")
prune.ln_structured(module.synthesis.L11_1044_51, "weight", amount=amount2, n=2, dim=1)
prune.remove(module.synthesis.L11_1044_51, "weight")
prune.ln_structured(module.synthesis.L13_1024_32, "weight", amount=amount3, n=2, dim=1)
prune.remove(module.synthesis.L13_1024_32, "weight")
prune.ln_structured(module.synthesis.L11_1044_51, "weight", amount=amount6, n=2, dim=1)
prune.remove(module.synthesis.L11_1044_51, "weight")

## ffhq r Layers
# prune.ln_structured(module.synthesis.L14_1024_3.affine, "weight", amount=amount0, n=2, dim=1)
# prune.remove(module.synthesis.L14_1024_3.affine, "weight")
# prune.ln_structured(module.synthesis.L1_36_1024.affine, "weight", amount=amount1, n=2, dim=1)
# prune.remove(module.synthesis.L1_36_1024.affine, "weight")
# prune.ln_structured(module.synthesis.L2_52_1024.affine, "weight", amount=amount2, n=2, dim=1)
# prune.remove(module.synthesis.L2_52_1024.affine, "weight")
# prune.ln_structured(module.synthesis.L3_52_1024.affine, "weight", amount=amount3, n=2, dim=1)
# prune.remove(module.synthesis.L3_52_1024.affine, "weight")
# prune.ln_structured(module.synthesis.L4_84_1024.affine, "weight", amount=amount4, n=2, dim=1)
# prune.remove(module.synthesis.L4_84_1024.affine, "weight")
# prune.ln_structured(module.synthesis.L5_148_1024, "weight", amount=amount5, n=2, dim=1)
# prune.remove(module.synthesis.L5_148_1024, "weight")
# prune.ln_structured(module.synthesis.L6_148_1024.affine, "weight", amount=amount6, n=2, dim=1)
# prune.remove(module.synthesis.L6_148_1024.affine, "weight")
# prune.ln_structured(module.synthesis.L7_276_645.affine, "weight", amount=amount4, n=2, dim=1)
# prune.remove(module.synthesis.L7_276_645.affine, "weight")
# prune.ln_structured(module.synthesis.L8_276_406.affine, "weight", amount=amount5, n=2, dim=1)
# prune.remove(module.synthesis.L8_276_406.affine, "weight")
# prune.ln_structured(module.synthesis.L9_532_256.affine, "weight", amount=amount6, n=2, dim=1)
# prune.remove(module.synthesis.L9_532_256.affine, "weight")

now = datetime.datetime.now().strftime("%H%M%S")

with open(f'yugioh-pruned-{pruning_strength}.pkl', 'wb') as f:
    pickle.dump(model, f)

    
# Generator                      Parameters  Buffers  Output shape           Datatype
# ---                            ---         ---      ---                    ---     
# mapping.fc0                    262656      -        [32, 512]              float32 
# mapping.fc1                    262656      -        [32, 512]              float32 
# mapping                        -           512      [32, 16, 512]          float32 
# synthesis.input.affine         2052        -        [32, 4]                float32 
# synthesis.input                1048576     3081     [32, 1024, 36, 36]     float32 
# synthesis.L0_36_1024.affine    525312      -        [32, 1024]             float32 
# synthesis.L0_36_1024           1049600     157      [32, 1024, 36, 36]     float32 
# synthesis.L1_36_1024.affine    525312      -        [32, 1024]             float32 
# synthesis.L1_36_1024           1049600     157      [32, 1024, 36, 36]     float32 
# synthesis.L2_52_1024.affine    525312      -        [32, 1024]             float32 
# synthesis.L2_52_1024           1049600     169      [32, 1024, 52, 52]     float32 
# synthesis.L3_52_1024.affine    525312      -        [32, 1024]             float32 
# synthesis.L3_52_1024           1049600     157      [32, 1024, 52, 52]     float32 
# synthesis.L4_84_1024.affine    525312      -        [32, 1024]             float32 
# synthesis.L4_84_1024           1049600     169      [32, 1024, 84, 84]     float32 
# synthesis.L5_148_1024.affine   525312      -        [32, 1024]             float32 
# synthesis.L5_148_1024          1049600     169      [32, 1024, 148, 148]   float16 
# synthesis.L6_148_1024.affine   525312      -        [32, 1024]             float32 
# synthesis.L6_148_1024          1049600     157      [32, 1024, 148, 148]   float16 
# synthesis.L7_276_645.affine    525312      -        [32, 1024]             float32 
# synthesis.L7_276_645           661125      169      [32, 645, 276, 276]    float16 
# synthesis.L8_276_406.affine    330885      -        [32, 645]              float32 
# synthesis.L8_276_406           262276      157      [32, 406, 276, 276]    float16 
# synthesis.L9_532_256.affine    208278      -        [32, 406]              float32 
# synthesis.L9_532_256           104192      169      [32, 256, 532, 532]    float16 
# synthesis.L10_1044_161.affine  131328      -        [32, 256]              float32 
# synthesis.L10_1044_161         41377       169      [32, 161, 1044, 1044]  float16 
# synthesis.L11_1044_102.affine  82593       -        [32, 161]              float32 
# synthesis.L11_1044_102         16524       157      [32, 102, 1044, 1044]  float16 
# synthesis.L12_1044_64.affine   52326       -        [32, 102]              float32 
# synthesis.L12_1044_64          6592        25       [32, 64, 1044, 1044]   float16 
# synthesis.L13_1024_64.affine   32832       -        [32, 64]               float32 
# synthesis.L13_1024_64          4160        25       [32, 64, 1024, 1024]   float16 
# synthesis.L14_1024_3.affine    32832       -        [32, 64]               float32 
# synthesis.L14_1024_3           195         1        [32, 3, 1024, 1024]    float16 
# synthesis                      -           -        [32, 3, 1024, 1024]    float32 
# ---                            ---         ---      ---                    ---     
# Total                          15093151    5600     -                      -       