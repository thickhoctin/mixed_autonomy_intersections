ckpt_path = 'results/fourway_1x1_penetration0.5_test/models/flow_700x700/model-0.pth'
new_ckpt_path = 'results/fourway_1x1_penetration0.5_test/models/flow_700x700/model-5.pth'

import torch
model_dict = torch.load(ckpt_path)
new_model_dict = dict(net=model_dict['net'])
torch.save(new_model_dict, new_ckpt_path)