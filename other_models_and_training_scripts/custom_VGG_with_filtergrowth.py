###MOdel prototype for gradual growth  FINAL
import torch
import torch.nn as nn



def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_filter_mask(stage_num, out_channels = 512, in_channels = 256, fr1 = 3, fr2 = 3, device=None):
    x = torch.zeros(out_channels,in_channels,fr1,fr2)
    if(stage_num==0):
    ##Stage 0
        x[:,:,fr1//2,fr2//2] = 1.0
    elif(stage_num==1):
    ##+ Stage 1
        for i in range(0,fr1,2):
            x[:,:,i,range(0,fr2,2)] = 1.0
        for i in range(1,fr2,2):
            x[:,:,range(1,fr1,2),i] = 1.0
    elif(stage_num>=2):
        x = torch.ones(out_channels,in_channels,fr1,fr2)
    #print("inp", device)
    x = x.to(device)
    #print("MASK DEVICE", x.device)
    return x

def check_device_for_tensor(tensor):
    cuda_check = tensor.is_cuda
    get_cuda_device = 'cpu'
    if cuda_check:
        get_cuda_device = tensor.get_device()
    return get_cuda_device

def modify_conv_layer_in_resnet_with_filter_mask(conv_layer_block, stage_num, is_inp_layer=False):
    #1. Check stage num
    if(stage_num>2):
        return None
    #2. Find all instances of conv2d in layer (for inp layer there), and for each:
        #3. Modify conv2d filter weights based on mask mutliplication (essentially make inactive positions 0)
    else:   
        if(is_inp_layer):
            None
        else:
            for layer in conv_layer_block.modules():
                #res_func = block.residual_function
                #print(res_func)
                #shortcut = block.shortcut
                #for layer in res_func:
                    #print(layer)
                if(isinstance(layer, torch.nn.Conv2d)):
                        #print('hi')
                    out_channels, in_channels, fr1, fr2 = layer.state_dict()['weight'].shape
                    device = layer.state_dict()['weight'].device
                        
                    filter_mask = get_filter_mask(stage_num, out_channels, in_channels, fr1, fr2, device)
                    updated_state_dict = layer.state_dict()
                    updated_state_dict['weight'] = torch.mul(layer.state_dict()['weight'], filter_mask)
                    layer.load_state_dict(updated_state_dict)
                    
                        
                        
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
        
    def forward(self, x, stage_num=3, change_at_inp_layer = True, filter_activation_method='change_fil'):
        
        if(filter_activation_method == 'change_fil'):
            if(self.birth_state):
                modify_conv_layer_in_customvgg_with_filter_mask(self.new_block, stage_num)
                x = self.new_block(x)
        #x = self.features(x)
                x = self.auxpool(x)
                x = torch.flatten(x, 1)
            #print(x[0:2][:3])
                x = self.classifier(x)
            else:
            #print('hi')
                modify_conv_layer_in_customvgg_with_filter_mask(self.prev_block, stage_num)
                existing_block_out = self.prev_block(x)
                modify_conv_layer_in_customvgg_with_filter_mask(self.new_block, stage_num)
                new_block_out = self.new_block(existing_block_out)
                existing_pooled = self.auxpool(existing_block_out)
                new_pooled = self.auxpool(new_block_out)
                existing_flatten = torch.flatten(existing_pooled, 1)
                #print(existing_flatten.shape)
                new_block_flatten = torch.flatten(new_pooled, 1)
                weighted_sum_out = self.alpha*new_block_flatten + (1-self.alpha)*existing_flatten
                
                x= self.classifier(weighted_sum_out)
            
        else:
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



        
        else: