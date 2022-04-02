import numpy as np
import torch
from torchvision import models
from torch import nn

class SelfTrainingModel(nn.Module):
    def __init__(self,
                 feature_extractor,
                 num_features,
                 num_classes=100,
                 predict_quality=True,
                 predict_correctness=True,
                 classifier_layers=[],
                 classifier_activation=None,
                 quality_layers=[],
                 quality_activation=None,
                 consider_classification_for_quality=False,
                 correctness_layers=[],
                 correctness_activation=None,
                 consider_classification_for_correctness=False):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.predict_quality = predict_quality
        self.predict_correctness = predict_correctness
        self.consider_classification_for_quality = consider_classification_for_quality
        self.consider_classification_for_correctness = consider_classification_for_correctness
        
        classifier_layers.insert(0, num_features)
        classifier_modules = []
        for l1, l2 in zip(classifier_layers[:-1], classifier_layers[1:]):
            classifier_modules.append(nn.Linear(in_features=l1, out_features=l2, bias=True))
            if classifier_activation != None:
                classifier_modules.append(classifier_activation())
        classifier_modules.append(nn.Linear(in_features=classifier_layers[-1], out_features=num_classes, bias=True))
        self.classifier = nn.Sequential(*classifier_modules)
        
        if self.predict_quality:
            quality_layers.insert(0, num_features + num_classes*self.consider_classification_for_quality)
            quality_modules = []
            for l1, l2 in zip(quality_layers[:-1], quality_layers[1:]):
                quality_modules.append(nn.Linear(in_features=l1, out_features=l2, bias=True))
                if quality_activation != None:
                    quality_modules.append(quality_activation())
            quality_modules.append(nn.Linear(in_features=quality_layers[-1], out_features=1, bias=True))
            self.quality_predictor = nn.Sequential(*quality_modules)
        
        if self.predict_correctness:
            correctness_layers.insert(0, num_features + num_classes + num_classes*self.consider_classification_for_correctness)
            correctness_modules = []
            for l1, l2 in zip(correctness_layers[:-1], correctness_layers[1:]):
                correctness_modules.append(nn.Linear(in_features=l1, out_features=l2, bias=True))
                if correctness_activation != None:
                    correctness_modules.append(correctness_activation())
            correctness_modules.append(nn.Linear(in_features=correctness_layers[-1], out_features=1, bias=True))
            self.correctness_predictor = nn.Sequential(*correctness_modules)
    
    def forward(self, x):
        if self.training and self.predict_correctness:
            image, label = x
        else:
            image = x
        features = self.feature_extractor(image)
        classification = self.classifier(features)
        if self.predict_quality:
            if self.consider_classification_for_quality:
                quality_input = torch.cat((features, classification))
            else:
                quality_input = features
            quality = self.quality_predictor(quality_input)
        else:
            quality = None
        if self.training and self.predict_correctness:
            if self.consider_classification_for_correctness:
                correctness_input = torch.cat((features, label, classification))
            else:
                correctness_input = torch.cat((features, label))
            correctness = self.correctness_predictor(correctness_input)
        else:
            correctness = None
        return (classification, quality, correctness)

def get_resnet_feature_extractor(num_blocks=50, pretrained=False, freeze_weights=False):
    num_blocks_to_constructor = {
        18:  models.resnet.resnet18,
        34:  models.resnet.resnet34,
        50:  models.resnet.resnet50,
        101: models.resnet.resnet101,
        152: models.resnet.resnet152}
    if not(num_blocks in num_blocks_to_constructor.keys()):
        raise ValueError('invalid num_blocks')
        
    model = num_blocks_to_constructor[num_blocks](pretrained=pretrained)
    if freeze_weights:
        for name, param in model.named_parameters():
            param.requires_grad = False
    
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    num_features = list(model.children())[-1].in_features
    return (feature_extractor, num_features)