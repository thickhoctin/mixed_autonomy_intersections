ckpt_path = 'results/fourway_1x1_penetration1_turn_adam_ppo_12.12/models/flow_700x700/model-175.pth'
new_ckpt_path = 'results/fourway_1x1_penetration0.5_turn_adam_ppo_14.12/models/flow_700x700/model-0.pth'

import torch
model_dict = torch.load(ckpt_path)
new_model_dict = dict(net=model_dict['net'])
torch.save(new_model_dict, new_ckpt_path)