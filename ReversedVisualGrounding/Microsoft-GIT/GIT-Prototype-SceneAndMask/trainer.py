from generativeimage2text.model import get_git_model
from generativeimage2text.tsv_io import load_from_yaml_file
from generativeimage2text.torch_common import torch_load
from generativeimage2text.torch_common import load_state_dict
from generativeimage2text.train import forward_backward
from transformers import BertTokenizer
import json
from azfuse import File

#Settings:
batch_size = 4
model_name = 'GIT_BASE'

param = {}
if File.isfile(f'aux_data/models/{model_name}/parameter.yaml'):
    param = load_from_yaml_file(f'aux_data/models/{model_name}/parameter.yaml')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model = get_git_model(tokenizer, param)
pretrained = f'output/{model_name}/snapshot/model.pt'
checkpoint = torch_load(pretrained)['model']
load_state_dict(model, checkpoint)
model.cuda()

# Calculate the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of trainable parameters:", trainable_params)

with open("ocid_data/OCID-For-GIT/train.json", "r") as file:
    train_data_points = json.load(file)
    
sample_counter = 1
delimiter = "o_o"
scene_files = []
mask_files = []
captions = []
for key, value in train_data_points.items():
    scene_file_path = value.get("scene_file_path")
    mask_file_path = value.get("mask_file_path")
    scene_instance_id = value.get("scene_instance_id")
    caption = value.get("caption")
    
    scene_files.append(scene_file_path)
    mask_files.append(str(scene_instance_id) + delimiter + mask_file_path)
    captions.append(caption)
    
    if(sample_counter%batch_size == 0):
        print("Loss:", forward_backward(model, scene_files, mask_files, captions).item())
        sample_counter = 0
        scene_files = []
        mask_files = []
        captions = []
    
    sample_counter += 1
    
if len(scene_files) != 0:
    print("Loss:", forward_backward(model, scene_files, mask_files, captions).item())
    sample_counter = 0
    scene_files = []
    mask_files = []
    captions = []