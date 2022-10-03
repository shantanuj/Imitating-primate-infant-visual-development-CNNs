###Custom VGG for ImageNet
import torch
import torch.nn as nn

class CustomVGG(nn.Module):
    def __init__(self, new_block, prev_block=None, prev_model_classifier=None, num_output_classes=1000, init_weights=False):
        super(CustomVGG, self).__init__()
        self.new_block = new_block
        self.auxpool = nn.AdaptiveAvgPool2d((7, 7)) #Try maxpool OR 1,1 pooling
        if(prev_model_classifier is None):
            self.classifier = nn.Sequential(
                nn.Linear(256 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_output_classes),
            )
        else:
            self.classifier = prev_model_classifier
        
        self.birth_state=True
        if(prev_block is not None):
            #print(prev_block)
            self.birth_state =False
            self.prev_block = prev_block
            self.post_stage_block = nn.Sequential(self.prev_block,self.new_block)
        else:
            self.prev_block = None
            self.post_stage_block = nn.Sequential(self.new_block)
        self.alpha = 0
        print(self.post_stage_block)
        print(self.classifier)
        #elif(prev_model_classifier)
        #if init_weights:
         #   self._initialize_weights()
        #if
        
    def forward(self, x):
        if(self.birth_state):
            x = self.new_block(x)
        #x = self.features(x)
            x = self.auxpool(x)
            x = torch.flatten(x, 1)
            #print(x[0:2][:3])
            x = self.classifier(x)
        else:
            #print('hi')
            existing_block_out = self.prev_block(x)
            new_block_out = self.new_block(existing_block_out)
            existing_pooled = self.auxpool(existing_block_out)
            new_pooled = self.auxpool(new_block_out)
            existing_flatten = torch.flatten(existing_pooled, 1)
            #print(existing_flatten.shape)
            new_block_flatten = torch.flatten(new_pooled, 1)
            #print(existing_flatten[0:2][:3])
            #print(new_block_flatten[0:2][:3])
            #print(new_block_flatten.shape)
            weighted_sum_out = self.alpha*new_block_flatten + (1-self.alpha)*existing_flatten
            #print(weighted_sum_out[0:2][:3])
            x= self.classifier(weighted_sum_out)
        return x
    
def make_layers(cfg, start_block=0, end_block = 5, batch_norm=False):
    layers = []
    in_channels = 3
    num_block=0
    #print(start_block)
    for i in range(0, end_block):
        #if v == 'M':
        #print(i)
        for layer_n in cfg[i]:
            if(layer_n=='M'):
                if(i in range(start_block,end_block)):
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            #num_block+=1
            #if(num_block>=cut_at_block):
             #   break
            else:
                conv2d = nn.Conv2d(in_channels, layer_n, kernel_size=3, padding=1)
                if batch_norm:
                    if(i in range(start_block,end_block)):
                        layers += [conv2d, nn.BatchNorm2d(layer_n), nn.ReLU(inplace=True)]
                else:
                    if(i in range(start_block,end_block)):
                        layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = layer_n
            
    return nn.Sequential(*layers)

cfgs_custom = {
    'A': [[256, 'M'],  #8+3 =11
          [256, 'M'], 
          [256, 256, 'M'],
          [256, 256, 'M'],
          [256, 256, 'M']],
    'B': [[256, 256, 'M'], #10+3 = 13
           [256, 256, 'M'], 
          [256, 256, 'M'], 
          [256, 256, 'M'],
          [256, 256, 'M']],
    'D': [[256, 256, 'M'], #13+3 = 16
          [256, 256, 'M'],
          [256, 256, 256, 'M'], 
          [256, 256, 256, 'M'],
          [256, 256, 256, 'M']],
    'E': [[256, 256, 'M'], #16+3 = 19
          [256, 256, 'M'],
          [256, 256, 256, 256, 'M'],
          [256, 256, 256, 256, 'M'],
          [256, 256, 256, 256, 'M']]
}

def make_custom_vgg(cfg, batch_norm, start_block =0, end_block =5, prev_block=None, prev_model_classifier = None, num_output_classes=1000, **kwargs):
    #if pretrained:
     #   kwargs['init_weights'] = False
    model = CustomVGG(new_block = make_layers(cfgs_custom[cfg], start_block= start_block, end_block = end_block, batch_norm=batch_norm), 
                      prev_block = prev_block,
                      prev_model_classifier= prev_model_classifier,
                      num_output_classes=num_output_classes,
                      init_weights=False,
                       **kwargs)
    return model
