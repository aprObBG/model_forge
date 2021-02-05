# -*- coding: utf-8 -*-
import torch
import numpy               as np
import torch.nn.functional as F
import quantize            as quant

class QuantLayer(torch.nn.Module):
    def __init__(self, freezeq=False, config=''):
        super(QuantLayer, self).__init__()
        self.config = config
        self.freezeq = freezeq

    # def forward(self, x):
    #     #import ipdb as pdb; pdb.set_trace()
    #     if self.config["quantization"].lower() == "int":
    #         return quant.int_nn(x, self.config["activation_i_width"])
    #     elif self.config["quantization"].lower() == "fixed":
    #         return quant.fixed_nn(x, self.config["activation_i_width"], self.config["activation_f_width"])
    #     elif self.config["quantization"].lower() == "bnn":
    #         # import ipdb as pdb; pdb.set_trace()
    #         return quant.bnn_sign(x)
    #     else:
    #         return x

    def forward(self, x):
        if (self.freezeq):
          return quant.fixed_nn(x, self.config["save_activation_i_width"], self.config["save_activation_f_width"])
        elif (self.config["quantization"].lower() == "fixed"):
          return quant.fixed_nn(x, self.config["activation_i_width"], self.config["activation_f_width"])
        else:
           return x
       # if self.config["quantization"].lower() == "fixed":
       #     return quant.fixed_nn(x, self.config["activation_i_width"], self.config["activation_f_width"])
       # else:
       #     return x
       #     # return quant.fixed_nn(x, self.config["activation_i_width"], self.config["activation_f_width"])

class ConvTranspose2dQuant(torch.nn.ConvTranspose3d):
    """
    Convolution layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       H            = 1.0,
                       W_LR_scale   = "Glorot",
                       config       = ""):
        #
        # Fan-in/fan-out computation
        #
        num_inputs   = in_channels
        num_units    = out_channels
        cnt = 0
        self.config  = config

        for x in kernel_size:
            num_inputs *= x
            num_units  *= x
        
        self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        
        self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, x):
        # self.cnt += 1
        # if self.cnt==1000:
        #import ipdb as pdb; pdb.set_trace()
        # if self.config["quantization"].lower() == "bnn":
        #     Wb = quant.bnn_sign(self.weight/self.H)*self.H
        # elif self.config["quantization"].lower() == "int":
        #     Wb = quant.int_nn(self.weight, self.config["weight_i_width"])
        # elif self.config["quantization"].lower() == "fixed":
        #     Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
        # elif self.config["quantization"].lower() == "ternary":
        #     Wb = quant.ternary_q(self.weight)
        # else:
        #     Wb = self.weight
       if self.config["quantization"].lower() == "fixed":
            Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
       else:
            Wb = self.weight
       return F.conv_transpose2d(x, Wb, self.bias, self.stride, self.padding)


class ConvTranspose3dQuant(torch.nn.ConvTranspose3d):
    """
    Convolution layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       H            = 1.0,
                       W_LR_scale   = "Glorot",
                       config       = ""):
        #
        # Fan-in/fan-out computation
        #
        num_inputs   = in_channels
        num_units    = out_channels
        cnt = 0
        self.config  = config

        for x in kernel_size:
            num_inputs *= x
            num_units  *= x
        
        self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        
        self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, x):
        # self.cnt += 1
        # if self.cnt==1000:
        #import ipdb as pdb; pdb.set_trace()
        # if self.config["quantization"].lower() == "bnn":
        #     Wb = quant.bnn_sign(self.weight/self.H)*self.H
        # elif self.config["quantization"].lower() == "int":
        #     Wb = quant.int_nn(self.weight, self.config["weight_i_width"])
        # elif self.config["quantization"].lower() == "fixed":
        #     Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
        # elif self.config["quantization"].lower() == "ternary":
        #     Wb = quant.ternary_q(self.weight)
        # else:
        #     Wb = self.weight
       if self.config["quantization"].lower() == "fixed":
            Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
       else:
            Wb = self.weight
       return F.conv_transpose3d(x, Wb, self.bias, self.stride, self.padding)


class Conv3dQuant(torch.nn.Conv3d):
    """
    Convolution layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       H            = 1.0,
                       W_LR_scale   = "Glorot",
                       config       = ""):
        #
        # Fan-in/fan-out computation
        #
        num_inputs   = in_channels
        num_units    = out_channels
        cnt = 0
        self.config  = config

        for x in kernel_size:
            num_inputs *= x
            num_units  *= x
        
        if H == "Glorot":
            self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        else:
            self.H          = H
        
        if W_LR_scale == "Glorot":
            self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        else:
            self.W_LR_scale = self.H
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, x):
        # self.cnt += 1
        # if self.cnt==1000:
        #import ipdb as pdb; pdb.set_trace()
        if self.config["quantization"].lower() == "bnn":
            Wb = quant.bnn_sign(self.weight/self.H)*self.H
        elif self.config["quantization"].lower() == "int":
            Wb = quant.int_nn(self.weight, self.config["weight_i_width"])
        elif self.config["quantization"].lower() == "fixed":
            Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
        elif self.config["quantization"].lower() == "ternary":
            Wb = quant.ternary_q(self.weight)
        else:
            Wb = self.weight
        return F.conv3d(x, Wb, self.bias, self.stride, self.padding, self.dilation, self.groups)

#
# PyTorch Convolution Layers
#

class Conv2dQuant(torch.nn.Conv2d):
    """
    Convolution layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       H            = 1.0,
                       W_LR_scale   = "Glorot",
                       freezeq       = False,
                       config       = ""):
        #
        # Fan-in/fan-out computation
        #
        num_inputs   = in_channels
        num_units    = out_channels
        cnt = 0
        self.config  = config
        self.freezeq = freezeq

        if (type(kernel_size) == int):
            num_inputs *= kernel_size
            num_inputs *= kernel_size
            num_units  *= kernel_size
            num_units  *= kernel_size
        else:
            for x in kernel_size:
                num_inputs *= x
                num_units  *= x
        
        self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        # if H == "Glorot":
        #     self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        # else:
        #     self.H          = H
        # 
        # if W_LR_scale == "Glorot":
        #     self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        # else:
        #     self.W_LR_scale = self.H
        
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, x):
        # self.cnt += 1
        # if self.cnt==1000:
        #import ipdb as pdb; pdb.set_trace()
        # if self.config["quantization"].lower() == "bnn":
        #     Wb = quant.bnn_sign(self.weight/self.H)*self.H
        # elif self.config["quantization"].lower() == "int":
        #     Wb = quant.int_nn(self.weight, self.config["weight_i_width"])
        # elif self.config["quantization"].lower() == "fixed":
        #     Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
        # elif self.config["quantization"].lower() == "ternary":
        #     Wb = quant.ternary_q(self.weight)
        # else:
        #     Wb = self.weight
        if (self.freezeq):
          Wb = quant.fixed_nn(self.weight, self.config["save_weight_i_width"], self.config["save_weight_f_width"])
        elif (self.config["quantization"].lower() == "fixed"):
          Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
        else:
          Wb = self.weight
        return F.conv2d(x, Wb, self.bias, self.stride, self.padding, self.dilation, self.groups)




#
# PyTorch Dense Layers
#

class LinearQuant(torch.nn.Linear):
    """
    Linear/Dense layer for BinaryNet.
    """
    
    def __init__(self, in_channels,
                       out_channels,
                       bias         = True,
                       H            = 1.0,
                       W_LR_scale   = "Glorot",
                       freezeq      = False,
                       config       = ""):
        #
        # Fan-in/fan-out computation
        #
        num_inputs   = in_channels
        num_units    = out_channels
        self.config       = config
        self.freezeq      = freezeq
        
        self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        # if H == "Glorot":
        #     self.H          = float(np.sqrt(1.5/(num_inputs + num_units)))
        # else:
        #     self.H          = H
        # 
        # if W_LR_scale == "Glorot":
        #     self.W_LR_scale = float(np.sqrt(1.5/(num_inputs + num_units)))
        # else:
        #     self.W_LR_scale = self.H
        
        super().__init__(in_channels, out_channels, bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weight.data.uniform_(-self.H, +self.H)
        if isinstance(self.bias, torch.nn.Parameter):
            self.bias.data.zero_()
    
    def constrain(self):
        self.weight.data.clamp_(-self.H, +self.H)
    
    def forward(self, input):
        #import ipdb as pdb; pdb.set_trace()
        # if self.config["quantization"] == "BNN":
        #     Wb = quant.bnn_sign(self.weight/self.H)*self.H
        #     return quant.bnn_sign(F.linear(input, Wb, self.bias))
        # elif self.config["quantization"] == "INT":
        #     Wb = quant.int_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
        #     return quant.int_nn(F.linear(input, Wb, self.bias), self.config["activation_i_width"], self.config["activation_f_width"])
        # else:
        #     Wb = self.weight
        #     return F.linear(input, Wb, self.bias)
        if (self.freezeq):
          Wb = quant.fixed_nn(self.weight, self.config["save_weight_i_width"], self.config["save_weight_f_width"])
        elif (self.config["quantization"].lower() == "fixed"):
          Wb = quant.fixed_nn(self.weight, self.config["weight_i_width"], self.config["weight_f_width"])
        else:
          Wb = self.weight

        return F.linear(input, Wb, self.bias)

