import json
from easydict import EasyDict
from pprint import pprint

def get_config_from_json(json_file):

    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = EasyDict(config_dict)

            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

def process_config(json_file):

    config, config_dict = get_config_from_json(json_file)

    try:
        print("")
        print(" *************************************** ")
        print("")
        print("The experiment name is {}".format(config.exp_agent))
        print("")
        pprint(config)
        print("")
        print("")
        print(" *************************************** ")
        print("")
    except AttributeError:
        print("ERROR!!..Please provide the agent name in json file..")
        exit(-1)

    return config_dict
