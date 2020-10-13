from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
sig = nn.Sigmoid()

class CamExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        
        for module_pos, module in self.model._modules.items():
            x = module(x)

            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)

                conv_output = x  

                return conv_output, x

    def forward_pass(self, x):
        conv_output, x = self.forward_pass_on_convolutions(x)
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = torch.flatten(out, 1)
        model_output = self.model.classifier(out)
        model_output = sig(model_output)
        # model_output = sig(model_output)
        return conv_output, model_output


class GradCam():
    def __init__(self, model, target_layer='features'):
        self.model = model

        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):

        input_image = input_image.unsqueeze(0).cuda()
        preds_list = []
        cam_list = []
        conv_output, model_output = self.extractor.forward_pass(input_image)
        conv_output = conv_output.squeeze(0)
        model_output_squeeze = model_output.squeeze(0)

        #preds
        for i in range(int(len(model_output_squeeze))):
            preds_list.append(model_output_squeeze[i].item())
        


        #cam
        for i in range(len(model_output_squeeze.cpu().data.numpy())):
            # target_class = model_output_squeeze.cpu().data.numpy()[i]

            one_hot_output = torch.FloatTensor(1, model_output_squeeze.size()[-1]).zero_().cuda()
            one_hot_output[0][i] = 1

            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
            
            model_output.backward(gradient=one_hot_output, retain_graph=True)

            guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]

            target = conv_output.cpu().data.numpy()

            weights = np.mean(guided_gradients, axis=(1,2))

            cam = np.ones(target.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  
            cam = np.uint8(cam * 255)  
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                        input_image.shape[3]), Image.ANTIALIAS))/255
        
            cam_list.append(cam)

        return cam_list, preds_list



