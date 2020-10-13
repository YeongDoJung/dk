import argparse
import json

from easydict import EasyDict

from utils.config import *
from agents import *

def main():
   
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--json_path', type=str, default='')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--generate_num', type=str, default='100')
    
    config = parser.parse_args()

    with open(config.json_path, "r") as read_file:
        data = json.load(read_file)

    config_dic = vars(config)
    config_dic.update(data)

    config = EasyDict(config_dic)

    agent_class = globals()[config.exp_agent]
    agent = agent_class(config)

    if config.mode == "run": #train + val + test
        agent.run()
    elif config.mode == "train":
        agent.train()
    elif config.mode == "test":
        agent.test()
    elif config.mode == "visual":
        agent.visual()
    else:
        print("config mode do not setting")



if __name__ == '__main__':
    main()

