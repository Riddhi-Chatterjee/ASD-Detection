'''
OCID-For-GIT contains three files: 

* train.json
* val.json
* test.json
Each contains samples in the following format:
{
    "<sentence_id_1>" : 
                {
                    "scene_file_path" : "<scene_file_path>",
                    "mask_file_path" : "<mask_file_path>",
                    "scene_instance_id" : "<scene_instance_id>",
                    "caption" : "<caption>"
                }
    "<sentence_id_2>" : 
                {
                    "scene_file_path" : "<scene_file_path>",
                    "mask_file_path" : "<mask_file_path>",
                    "scene_instance_id" : "<scene_instance_id>",
                    "caption" : "<caption>"
                }
    ...
}
'''

import json
from os.path import exists
import os

if(exists("ocid_data/OCID-For-GIT")):
    os.system("rm -r ocid_data/OCID-For-GIT")
    
os.system("mkdir ocid_data/OCID-For-GIT")

for split in ["train", "val", "test"]:
    samples = {}
    
    # Load the data from the JSON file
    with open("ocid_data/OCID-Ref/"+ split + "_expressions.json", "r") as file:
        data = json.load(file)

    # Iterate over each key in the dictionary and extract the required information
    for key, value in data.items():
        take_id = value.get("take_id")
        scene_path = "ocid_data/OCID-dataset/" + value.get("scene_path")
        sequence_path = "ocid_data/OCID-dataset/" + value.get("sequence_path")
        scene_instance_id = value.get("scene_instance_id")
        sentence = value.get("sentence")

        mask_path = sequence_path + "/label/" + scene_path.split("/")[-1]
        
        # Create a new data point JSON object
        data_point = {
            "scene_file_path": scene_path,
            "mask_file_path": mask_path,
            "scene_instance_id": scene_instance_id,
            "caption": sentence
        }

        # Add the data point to the dictionary
        samples[key] = data_point
        
    # Dump the data points into a JSON file
    with open("ocid_data/OCID-For-GIT/" + split + ".json", "w") as outfile:
        json.dump(samples, outfile, indent=4)
