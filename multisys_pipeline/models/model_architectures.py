import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
#from torch.nn import functional as F# https://twitter.com/francoisfleuret/status/1247576431762210816
from torch import nn
#device = torch.device('cpu')# device = torch.device('cuda')
import warnings

trunc_window = 10 # for tBPTT quick experiments 

## Dale's Law and sparsity helpers     
prop_e = 0.8 # proportion excitatory 

def effective_weight(weight, mask):
    return torch.abs(weight) * mask 

def effective_linear(z, weight, mask, bias=None):
    Weff = effective_weight(weight, mask)
    return torch.nn.functional.linear(z, Weff, bias=bias)

def effective_weight_sparsity(weight, mask): # only constrain sparsity, no Dale constraint 
    return weight * mask 

def effective_linear_sparsity(z, weight, mask, bias=None):
    Weff = effective_weight_sparsity(weight, mask)
    return torch.nn.functional.linear(z, Weff, bias=bias)

def get_mask(n_recurrent):
    e_size = int(prop_e * n_recurrent)
    i_size = n_recurrent - e_size 
    mask = np.tile([1]*e_size+[-1]*i_size, (n_recurrent, 1))
    np.fill_diagonal(mask, 0)
    return torch.tensor(mask, dtype=torch.float32)

def get_sparsity_mask(n_recurrent, p):
    n_out = n_in = n_recurrent
    W_adj = np.random.rand(n_out, n_in) < p
    mask = torch.tensor(W_adj, dtype=torch.float32)
    return mask 
    

#%%##############################################################################
# continuous time recurrent neural network
# Tau * dah/dt = -ah + Wahh @ f(ah) + Wahx @ x + bah
#
# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)􏰩    
# h[t] = f(ah[t]) + activity_noise[t], if t > 0
# y[t] = Wyh @ h[t] + by  output

# parameters to be learned: Wahh, Wahx, Wyh, bah, by, ah0(optional). In this implementation h[0] = f(ah[0]) with no noise added to h[0] except potentially through ah[0]
# constants that are not learned: dt, Tau, activity_noise
# Equation 1 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
class CTRNN(nn.Module):# class CTRNN inherits from class torch.nn.Module
    def __init__(self, n_input, n_recurrent, n_output, Wahx=None, Wahh=None, Wyh=None, bah=None, by=None, activation_function='retanh', ah0=None, LEARN_ah0=False, LEARN_Wahx=True, LEARN_Wahh=True, LEARN_bah=True, LEARN_OUTPUTWEIGHT=True, LEARN_OUTPUTBIAS=True, dt=1, Tau=10, gain_Wh2h=None, learning_mode=0, dale_constraint=False, conn_density=-1):
        super().__init__()# super allows you to call methods of the superclass in your subclass
        self.fc_x2ah = nn.Linear(n_input, n_recurrent)# Wahx @ x + bah
        self.fc_h2ah = nn.Linear(n_recurrent, n_recurrent, bias = False)# Wahh @ h
        self.fc_h2y = nn.Linear(n_recurrent, n_output)# y = Wyh @ h + by
        self.n_parameters = n_recurrent**2 + n_recurrent*n_input + n_recurrent + n_output*n_recurrent + n_output# number of learned parameters in model
        self.dt = dt
        self.Tau = torch.tensor(Tau, dtype=torch.float32)
        self.learning_mode = learning_mode 
        self.dale_constraint = dale_constraint 
        self.mask = get_mask(n_recurrent)
        if 0 < conn_density < 1:
            if self.dale_constraint:
                sparsity_mask = get_sparsity_mask(n_recurrent, conn_density)
                self.mask *= sparsity_mask
            else:
                self.mask = get_sparsity_mask(n_recurrent, conn_density)
            
        #------------------------------
        # initialize the biases bah and by 
        if bah is not None:
            self.fc_x2ah.bias = torch.nn.Parameter(torch.squeeze(bah))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        if by is not None:
            self.fc_h2y.bias = torch.nn.Parameter(torch.squeeze(by))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        #------------------------------
        if LEARN_bah==False:# this must go after the line self.fc_x2ah.bias = torch.nn.Parameter(torch.squeeze(bah)) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_bah = False then bah does not change during gradient descent learning
            self.fc_x2ah.bias.requires_grad = False# Wahx @ x + bah
            self.n_parameters = self.n_parameters - n_recurrent# number of learned parameters in model
        if LEARN_OUTPUTBIAS==False:# this must go after the line self.fc_h2y.bias = torch.nn.Parameter(torch.squeeze(by)) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_OUTPUTBIAS = False then by does not change during gradient descent learning
            self.fc_h2y.bias.requires_grad = False# y = Wyh @ h + by
            self.n_parameters = self.n_parameters - n_output# number of learned parameters in model
        #------------------------------
        # initialize input(Wahx), recurrent(Wahh), output(Wyh) weights 
        if Wahx is not None:
            self.fc_x2ah.weight = torch.nn.Parameter(Wahx)# Wahx @ x + bah
        if Wahh is not None:
            self.fc_h2ah.weight = torch.nn.Parameter(Wahh)# Wahh @ h
        if Wyh is not None:
            self.fc_h2y.weight = torch.nn.Parameter(Wyh)# y = Wyh @ h + by
        #------------------------------
        if LEARN_Wahx==False:# this must go after the line self.fc_x2ah.weight = torch.nn.Parameter(Wahx) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_Wahx = False then Wahx does not change during gradient descent learning
            self.fc_x2ah.weight.requires_grad = False# Wahx @ x + bah
            self.n_parameters = self.n_parameters - n_recurrent*n_input# number of learned parameters in model
        if LEARN_Wahh==False:# this must go after the line self.fc_h2ah.weight = torch.nn.Parameter(Wahh) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_Wahh = False then Wahh does not change during gradient descent learning
            self.fc_h2ah.weight.requires_grad = False# Wahh @ h
            self.n_parameters = self.n_parameters - n_recurrent*n_recurrent# number of learned parameters in model
        if LEARN_OUTPUTWEIGHT==False:# this must go after the line self.fc_h2y.weight = torch.nn.Parameter(Wyh) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_OUTPUTWEIGHT = False then Wyh does not change during gradient descent learning
            self.fc_h2y.weight.requires_grad = False# y = Wyh @ h + by
            self.n_parameters = self.n_parameters - n_output*n_recurrent# number of learned parameters in model
        #------------------------------
        # set the activation function for h 
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        #self.activation_function = lambda x: f(x, activation_function)
        self.activation_function = activation_function
        #------------------------------
        # set the initial state ah0
        if ah0 is None:
            self.ah0 = torch.nn.Parameter(torch.zeros(n_recurrent), requires_grad=False)# (n_recurrent,) tensor
        else:
            self.ah0 = torch.nn.Parameter(ah0, requires_grad=False)# (n_recurrent,) tensor
        if LEARN_ah0:
            #self.ah0 = self.ah0.requires_grad=True# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.ah0 = torch.nn.Parameter(self.ah0, requires_grad=True)# (n_recurrent,) tensor
            self.n_parameters = self.n_parameters + n_recurrent# number of learned parameters in model
        # Modulate the initialization gain 
        if gain_Wh2h is not None:
            self.fc_h2ah.weight.data.copy_(self.fc_h2ah.weight.data*gain_Wh2h) 
            
        if 0 < conn_density < 1: # rescale weight to the same norm as b4 sparsity 
            self.fc_h2ah.weight.data.copy_(self.fc_h2ah.weight.data/np.sqrt(conn_density)) 
            
        # ensure balanced initialization if dale_constraint 
        if dale_constraint:
            N_e = int(prop_e * n_recurrent)
            N_i = n_recurrent - N_e
            self.fc_h2ah.weight.data[:, :N_e] /= (N_e/N_i) 
        
    # output y for all n_T timesteps   
    def forward(self, model_input_forwardpass):# nn.Linear expects inputs of size (*, n_input) where * means any number of dimensions including none
        input = model_input_forwardpass['input']
        activity_noise = model_input_forwardpass['activity_noise']
        conn_density = model_input_forwardpass.get('conn_density', -1)
        if len(input.shape)==2:# if input has size (n_T, n_input) because there is only a single trial then add a singleton dimension
            input = input[None,:,:]# (n_trials, n_T, n_input)
            activity_noise = activity_noise[None,:,:]# (n_trials, n_T, n_recurrent)            
        
        dt = self.dt
        Tau = self.Tau
        n_trials, n_T, n_input = input.shape# METHOD 2
        ah = self.ah0.repeat(n_trials, 1)# (n_trials, n_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        h = compute_activation_function(ah, self.activation_function)# h0, this implementation doesn't add noise to h0
        hstore = []# (n_trials, n_T, n_recurrent)
        ystore = []
        for t in range(n_T):
            if self.learning_mode == 1: # e-prop
                if self.dale_constraint:
                    ah = ah + (dt/Tau) * (-ah + \
                                effective_linear(h.detach(), self.fc_h2ah.weight, self.mask) + self.fc_x2ah(input[:,t]))                   
                elif 0 < conn_density < 1:
                    ah = ah + (dt/Tau) * (-ah + \
                                effective_linear_sparsity(h.detach(), self.fc_h2ah.weight, self.mask, bias=self.fc_h2ah.bias) + self.fc_x2ah(input[:,t]))                    
                else:
                    ah = ah + (dt/Tau) * (-ah + self.fc_h2ah(h.detach()) + self.fc_x2ah(input[:,t]))# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)
            else:
                if self.learning_mode == -1: # tBPTT
                    if ((n_T - t) % trunc_window) == 0:
                        h = h.detach()
                
                if self.dale_constraint:
                    ah = ah + (dt/Tau) * (-ah + effective_linear(h, self.fc_h2ah.weight, self.mask) + self.fc_x2ah(input[:,t]))
                elif 0 < conn_density < 1:
                    ah = ah + (dt/Tau) * (-ah + effective_linear_sparsity(h, self.fc_h2ah.weight, self.mask, bias=self.fc_h2ah.bias) + self.fc_x2ah(input[:,t]))
                else:
                    ah = ah + (dt/Tau) * (-ah + self.fc_h2ah(h) + self.fc_x2ah(input[:,t]))# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + Wahh @ h[t−1] + 􏰨Wahx @ x[t] +  bah)
            h = compute_activation_function(ah, self.activation_function)  +  activity_noise[:,t,:]# activity_noise has shape (n_trials, n_T, n_recurrent) 
            hstore.append(h)# hstore += [h]
            yout = self.fc_h2y(h)
            ystore.append(yout)
        hstore = torch.stack(hstore,dim=1)# (n_trials, n_T, n_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (*, n_recurrent) where * means any number of dimensions including none
        ystore = torch.stack(ystore,dim=1) 
        model_output_forwardpass = {'output':ystore, 'activity':hstore} 
        return model_output_forwardpass


#%%#--------------------------------------------------------------------------
#               compute specified nonlinearity/activation function 
#-----------------------------------------------------------------------------
def compute_activation_function(IN,string,*args):# ags[0] is the slope for string='tanhwithslope'
    if string == 'linear':
        F = IN
        return F
    elif string == 'logistic':
        F = 1 / (1 + torch.exp(-IN))
        return F
    elif string == 'smoothReLU':# smoothReLU or softplus 
        F = torch.log(1 + torch.exp(IN))# always greater than zero  
        return F
    elif string == 'ReLU':# rectified linear units
        #F = torch.maximum(IN,torch.tensor(0))
        F = torch.clamp(IN, min=0)
        return F
    elif string == 'swish':# swish or SiLU (sigmoid linear unit)
        # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        # Elfwing et al. 2017 "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"
        # Ramachandran et al. 2017 "Searching for activation functions"
        sigmoid = 1/(1+torch.exp(-IN))
        F = torch.mul(IN,sigmoid)# x*sigmoid(x), torch.mul performs elementwise multiplication
        return F
    elif string == 'mish':# Misra 2019 "Mish: A Self Regularized Non-Monotonic Neural Activation Function
        F = torch.mul(IN, torch.tanh(torch.log(1+torch.exp(IN))))# torch.mul performs elementwise multiplication
        return F
    elif string == 'GELU':# Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        F = 0.5 * torch.mul(IN, (1 + torch.tanh(torch.sqrt(torch.tensor(2/np.pi))*(IN + 0.044715*IN**3))))# fast approximating used in original paper
        #F = x.*normcdf(x,0,1);% x.*normcdf(x,0,1)  =  x*0.5.*(1 + erf(x/sqrt(2)))
        #figure; hold on; x = linspace(-5,5,100); plot(x,x.*normcdf(x,0,1),'k-'); plot(x,0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3))),'r--')           
        return F
    elif string == 'ELU':# exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1
        inegativeIN = (IN < 0)
        F = IN.clone() 
        F[inegativeIN] = alpha * (torch.exp(IN[inegativeIN]) - 1) 
        return F
    elif string == 'tanh':
        F = torch.tanh(IN)
        return F
    elif string == 'tanhwithslope':
        a = args[0]
        F = torch.tanh(a*IN)# F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)  
        return F
    elif string == 'tanhlecun':# LeCun 1998 "Efficient Backprop" 
        F = 1.7159*torch.tanh(2/3*IN)# F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)  
        return F
    elif string == 'lineartanh':
        #F = torch.minimum(torch.maximum(IN,torch.tensor(-1)),torch.tensor(1))# -1(x<-1), x(-1<=x<=1), 1(x>1)
        F = torch.clamp(IN, min=-1, max=1)
        return F
    elif string == 'retanh':# rectified tanh
        F = torch.maximum(torch.tanh(IN),torch.tensor(0))
        return F
    elif string == 'binarymeanzero':# binary units with output values -1 and +1
        #F = (IN>=0) - (IN<0)# matlab code
        F = 1*(IN>=0) - 1*(IN<0)# multiplying by 1 converts True to 1 and False to 0
        return F
    else:
        print('Unknown transfer function type')
        


#-----------------------------------------------------------------------------
#    compute derivative of nonlinearity/activation function with respect to its input dF(x)/dx
#-----------------------------------------------------------------------------
def compute_activation_function_gradient(F,string,*args):# input has already been passed through activation function, F = f(x). ags[0] is the slope for string='tanhwithslope'
    if string == 'linear':
        dFdx = torch.ones(F.shape)
        return dFdx
    elif string == 'logistic':
        dFdx = F - F**2# dfdx = f(x)-f(x).^2 = F-F.^2
        return dFdx
    elif string == 'smoothReLU':# smoothReLU or softplus 
        dFdx = 1 - torch.exp(-F)# dFdx = 1./(1 + exp(-x)) = 1 - exp(-F)
        return dFdx
    elif string == 'ReLU':# rectified linear units
        dFdx = 1.0*(F > 0)# F > 0 is the same as x > 0 for ReLU nonlinearity, multiplying by 1 converts True to 1 and False to 0, multiplying by 1.0 versus 1 makes dFdx a float versus an integer
        return dFdx
    elif string == 'ELU':# exponential linear units, Clevert et al. 2015 "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
        alpha = 1
        inegativex = (F < 0)# F < 0 is the same as x < 0 for ELU nonlinearity
        dFdx = torch.ones(F.shape); dFdx[inegativex] = F[inegativex] + alpha
        return dFdx
    elif string == 'tanh':
        dFdx = 1 - F**2# dfdx = 1-f(x).^2 = 1-F.^2
        return dFdx
    elif string == 'tanhwithslope':
        a = args[0]
        dFdx = a - a*(F**2)# F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)  
        return dFdx
    elif string == 'tanhlecun':# LeCun 1998 "Efficient Backprop"
        dFdx = 1.7159*2/3 - 2/3*(F**2)/1.7159# F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)
        return dFdx
    elif string == 'lineartanh':
        dFdx = 1*((F>-1) * (F<1))# 0(F<=-1), 1(-1<F<1), 0(F>=1), not quite right at x=-1 and x=1, * is elementwise multiplication
        return dFdx
    elif string == 'retanh':# rectified tanh
        dFdx = (1 - F**2) * (F > 0)# dfdx = 1-f(x).^2 = 1-F.^2,  * is elementwise multiplication
        return dFdx
    elif string == 'binarymeanzero':# binary units with output values -1 and +1
        dFdx = torch.zeros(F.shape)
        return dFdx
    else:
        print('Unknown transfer function type')







    

    