import numpy as np# https://stackoverflow.com/questions/11788950/importing-numpy-into-functions
import torch
#from torch.nn import functional as F# https://twitter.com/francoisfleuret/status/1247576431762210816
from torch import nn
#device = torch.device('cpu')# device = torch.device('cuda')
import warnings

trunc_window = 10 # for tBPTT quick experiments 

#%%##############################################################################
# ModProp accessories 
mu = 0.2 # activation derivative approx const 
ntype_E = 8
ntype_I = 8
sparsity_prob = 0.5 # only at init 

class ModPropActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.relu(x) # x: check match with linear net BPTT

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output*mu # const activation derivative
    
class CustLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, Wrec, Wab):
        ctx.save_for_backward(z, Wrec, Wab)
        return torch.nn.functional.linear(z, Wrec)

    @staticmethod
    def backward(ctx, grad_out):
        z, Wrec, Wab = ctx.saved_tensors
        grad_in = torch.linalg.matmul(grad_out, Wab) # bk,kj->bj
        grad_Wr = torch.linalg.matmul(grad_out.t(), z) # jb,bi->ji
        return grad_in, grad_Wr, torch.zeros_like(Wab)
    
# for FA 
class CustLinear2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, W_, Wfb):
        ctx.save_for_backward(z, W_, Wfb)
        return torch.nn.functional.linear(z, W_)

    @staticmethod
    def backward(ctx, grad_out):
        z, W_, Wfb = ctx.saved_tensors
        grad_in = torch.einsum('bk,ki->bi', grad_out, Wfb) # bk,kj->bj
        grad_Wr = torch.einsum('bk,bi->ki', grad_out, z) # jb,bi->ji
        return grad_in, grad_Wr, torch.zeros_like(Wfb) 
    
## Dale's Law stuff     
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
    # # W_adj = np.ones((n_recurrent, n_recurrent)) # for debugging 
    # nb_non_zero = int(n_in * n_out * p)
    # W_adj = np.zeros((n_out, n_in), dtype=bool)
    # ind_in = np.random.choice(np.arange(n_in), size=nb_non_zero)
    # ind_out = np.random.choice(np.arange(n_out), size=nb_non_zero)
    # W_adj[ind_out, ind_in] = True
    mask = torch.tensor(W_adj, dtype=torch.float32)
    return mask 

def get_mask_2pop_wout(n_recurrent, n_out):
    N_per_pop = int(n_recurrent / 2)
    mask = np.zeros((n_out, n_recurrent))
    mask[:,N_per_pop:] = 1 # only the 2nd population reads out 
    return torch.tensor(mask, dtype=torch.float32)

def get_mask_2pop_win(n_in, n_recurrent):
    N_per_pop = int(n_recurrent / 2)
    mask = np.zeros((n_recurrent, n_in))
    mask[:N_per_pop,:] = 1 # only the first population reads in
    return torch.tensor(mask, dtype=torch.float32)

# block initialization 
def get_block_mask(n_recurrent, Wrec):
    N_e = int(prop_e * n_recurrent)
    N_i = n_recurrent - N_e
    n_per_type_I = int(N_i/ntype_I)
    n_per_type_E = int(N_e/ntype_E)
    if N_i % ntype_I:
        inh_idx = list(range(N_e, n_recurrent, n_per_type_I)[:-1])
    else:
        inh_idx = list(range(N_e, n_recurrent, n_per_type_I))
    if N_e % ntype_E:
        exc_idx = list(range(0, N_e, n_per_type_E)[:-1])
    else:
        exc_idx = list(range(0, N_e, n_per_type_E))
    inh_idx.append(n_recurrent)
    tp_idx_ = np.concatenate((np.array(exc_idx), np.array(inh_idx)))
    tp_idx = np.stack((tp_idx_[:-1], tp_idx_[1:]), axis=1)
    n_type = len(tp_idx)
    
    for ii in range(n_type):
        for jj in range(n_type):
            prob = sparsity_prob    # randomly set sparsity_prob% of the block to 0
            pp = int(np.random.uniform() <= prob)
            # pp = int(pp / prob)
            M_block = pp*torch.ones_like(Wrec[tp_idx[ii][0]:tp_idx[ii][1],tp_idx[jj][0]:tp_idx[jj][1]])
            if jj==0: # new row
                M_row = M_block
            else:
                M_row = torch.cat((M_row, M_block), dim=1)
            if jj==(n_type-1): # finished a row
                if ii==0:
                    mask_ = M_row
                else:
                    mask_ = torch.cat((mask_, M_row), dim=0)
    return mask_, tp_idx
                    
def get_Wab(Wrec, tp_idx):
    n_type = len(tp_idx)
    for ii in range(n_type):
        for jj in range(n_type):
            W_block = Wrec[tp_idx[ii][0]:tp_idx[ii][1],tp_idx[jj][0]:tp_idx[jj][1]].detach()
            Wav = torch.mean(W_block)
            if jj==0: # new row
                Wab_row = Wav * torch.ones_like(W_block)
            else:
                Wab_row = torch.cat((Wab_row, Wav * torch.ones_like(W_block)), dim=1)
            if jj==(n_type-1): # finished a row
                if ii==0:
                    Wab_ = Wab_row
                else:
                    Wab_ = torch.cat((Wab_, Wab_row), dim=0)
    return Wab_
    

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
        assert learning_mode != 2 or (activation_function == 'ReLU' and dale_constraint), \
        "If learning_mode is 2, then activation_function must be 'ReLU' and dale_constraint must be True"
        if learning_mode == 2:
            warnings.warn("Warning: current RNN implementation does not enforce Dale's Law. "
                          "As such, ModProp implementation runs without the cell-type approximation. "
                          "Future developments will incorporate Dale's Law and cell types.")
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
        #------------------------------
        #self.LEARN_ah0 = LEARN_ah0
        #if LEARN_ah0:
        #    self.ah0 = torch.nn.Parameter(torch.zeros(n_recurrent), requires_grad=True)# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
        #    self.n_parameters = self.n_parameters + n_recurrent# number of learned parameters in model
        #------------------------------
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
            
        # if ReLU + dale, then assume ModProp run, then init block for all rules for comparison 
        if (activation_function == 'ReLU' and dale_constraint):
            self.mask_h2h, self.tp_idx = get_block_mask(n_recurrent, self.fc_h2ah.weight.data)
            self.fc_h2ah.weight.data.copy_(self.fc_h2ah.weight.data*self.mask_h2h) 
            self.Wab = get_Wab(effective_weight(self.fc_h2ah.weight, self.mask), self.tp_idx)
            
            # code to further balance connection strength for each neuron  
            W_ = self.fc_h2ah.weight.data 
            N_e = int(prop_e * n_recurrent)
            W_[:, :N_e] = torch.abs(W_[:, :N_e])  # Excitatory
            W_[:, N_e:] = -torch.abs(W_[:, N_e:]) # Inhibitory
            for neuron in range(n_recurrent):
                sum_abs_weights = torch.sum(torch.abs(W_[neuron])) 
                W_[neuron] /= sum_abs_weights
            self.fc_h2ah.weight.data.copy_(W_)   
            
        # FA
        if learning_mode == -2:
            self.Wfb = torch.empty_like(self.fc_h2y.weight)  # Creates an empty tensor with the same shape and data type
            bound = 1 / np.sqrt(n_recurrent)
            nn.init.uniform_(self.Wfb, -bound, bound)  # Initialize with a uniform distribution
            self.Wfb.requires_grad = False
            
        # self.mask_in = get_mask_2pop_win(n_input, n_recurrent) # TEMP: TWO POP
        # self.mask_out = get_mask_2pop_wout(n_recurrent, n_output)
        
        
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
        #n_trials, n_T, n_input = input.size()# METHOD 1
        n_trials, n_T, n_input = input.shape# METHOD 2
        ah = self.ah0.repeat(n_trials, 1)# (n_trials, n_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        #if self.LEARN_ah0:
        #    ah = self.ah0.repeat(n_trials, 1)# (n_trials, n_recurrent) tensor, all trials should have the same initial value for h, not different values for each trial
        #else:
        #    ah = input.new_zeros(n_trials, n_recurrent)# tensor.new_zeros(size) returns a tensor of size size filled with 0. By default, the returned tensor has the same torch.dtype and torch.device as this tensor. 
        #h = self.activation_function(ah)# h0
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
                    # ah = ah + (dt/Tau) * (-ah + self.fc_h2ah(h.detach()) + effective_linear_sparsity(input[:,t], self.fc_x2ah.weight, self.mask_in, bias=self.fc_x2ah.bias)) # TEMP: TWO POP
            elif self.learning_mode == 2: # ModProp    
                # activation derivative approx is only applied b4 the "Linear" layer, not at ah->h->readout 
                h_ = ModPropActivation.apply(ah)  
                if t > 0:
                    h_ += activity_noise[:,t-1,:]
                
                if self.dale_constraint:
                    ah = ah + (dt/Tau) * (-ah + \
                            CustLinear.apply(h_, effective_weight(self.fc_h2ah.weight, self.mask), self.Wab) + self.fc_x2ah(input[:,t]))                    
                elif 0 < conn_density < 1:
                    ah = ah + (dt/Tau) * (-ah + \
                            CustLinear.apply(h_, effective_weight_sparsity(self.fc_h2ah.weight, self.mask), self.Wab) + self.fc_x2ah(input[:,t]))   
                else:                     
                    ah = ah + (dt/Tau) * (-ah + CustLinear.apply(h_, self.fc_h2ah.weight, self.Wab) + self.fc_x2ah(input[:,t]))         
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
                    # ah = ah + (dt/Tau) * (-ah + self.fc_h2ah(h) + effective_linear_sparsity(input[:,t], self.fc_x2ah.weight, self.mask_in, bias=self.fc_x2ah.bias)) # TEMP: TWO POP
            #h = self.activation_function(ah)  +  activity_noise[:,t,:]# activity_noise has shape (n_trials, n_T, n_recurrent) 
            h = compute_activation_function(ah, self.activation_function)  +  activity_noise[:,t,:]# activity_noise has shape (n_trials, n_T, n_recurrent) 
            hstore.append(h)# hstore += [h]
            # ystore.append(effective_linear_sparsity(h, self.fc_h2y.weight, self.mask_out, bias=self.fc_h2y.bias)) #TEMP: TWO POP
            if self.learning_mode == -2:
                yout = CustLinear2.apply(h, self.fc_h2y.weight, self.Wfb) + self.fc_h2y.bias
            else:
                yout = self.fc_h2y(h)
            ystore.append(yout)
        hstore = torch.stack(hstore,dim=1)# (n_trials, n_T, n_recurrent), each appended h is stored in hstore[:,i,:], nn.Linear expects inputs of size (*, n_recurrent) where * means any number of dimensions including none
        ystore = torch.stack(ystore,dim=1) # TEMP: TWO POP
        # #return self.fc_h2y(hstore), hstore# (n_trials, n_T, n_output/n_recurrent) tensor, y = Wyh @ h + by
        # model_output_forwardpass = {'output':self.fc_h2y(hstore), 'activity':hstore}# (n_trials, n_T, n_output/n_recurrent) tensor, y = Wyh @ h + by
        model_output_forwardpass = {'output':ystore, 'activity':hstore} # TEMP: TWO POP
        return model_output_forwardpass



'''    
# A note on broadcasting:
# multiplying a (N,) array by a (M,N) matrix with * will broadcast element-wise
torch.manual_seed(123)# set random seed for reproducible results  
n_trials = 2  
Tau = torch.randn(5); Tau[-1] = 10
ah = torch.randn(n_trials,5)
A = ah + 1/Tau * (-ah)
A_check = -700*torch.ones(n_trials,5)
for i in range(n_trials):
    A_check[i,:] = ah[i,:] + 1/Tau * (-ah[i,:])# * performs elementwise multiplication
print(f"Do A and A_check have the same shape and are element-wise equal within a tolerance? {A.shape == A_check.shape and np.allclose(A, A_check)}")
'''


#%%##############################################################################
# low pass continuous time recurrent neural network
# Tau * dr/dt = -r + f(Wrr @ r + Wrx @ x + br) 
#
# r[t] = r[t-1] + (dt/Tau) * (-r[t-1] + f(Wrr @ r[t-1] + Wrx @ x[t] + br)  +  brneverlearn[t])
# y[t] = Wyr @ r[t] + by  output

# parameters to be learned: Wrr, Wrx, Wyr, br, by, r0(optional)
# constants that are not learned: dt, Tau, brneverlearn
# Equation 2 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
# "Note that equation 2 can be written Tau*dr/dt = -r + f(v). That is, if we regard v as a voltage 
# and f(v) as a firing rate, as suggested by the "derivation" in the appendix, then r is a low-pass-filtered version of the firing rate"
class LowPassCTRNN(nn.Module):# class LowPassCTRNN inherits from class torch.nn.Module
    def __init__(self, n_input, n_recurrent, n_output, Wrx=None, Wrr=None, Wyr=None, br=None, by=None, activation_function='retanh', r0=None, LEARN_r0=False, LEARN_OUTPUTWEIGHT=True, LEARN_OUTPUTBIAS=True, dt=1, Tau=10):
        super().__init__()# super allows you to call methods of the superclass in your subclass
        self.fc_x2r = nn.Linear(n_input, n_recurrent)# Wrx @ x + br
        self.fc_r2r = nn.Linear(n_recurrent, n_recurrent, bias = False)# Wrr @ r
        self.fc_r2y = nn.Linear(n_recurrent, n_output)# y = Wyr @ r + by
        self.n_parameters = n_recurrent**2 + n_recurrent*n_input + n_recurrent + n_output*n_recurrent + n_output# number of learned parameters in model
        self.dt = dt
        self.Tau = Tau
        #------------------------------
        # initialize the biases br and by 
        if br is not None:
            self.fc_x2r.bias = torch.nn.Parameter(torch.squeeze(br))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        if by is not None:
            self.fc_r2y.bias = torch.nn.Parameter(torch.squeeze(by))# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48-L52
        #------------------------------
        if LEARN_OUTPUTBIAS==False:# this must go after the line self.fc_r2y.bias = torch.nn.Parameter(torch.squeeze(by)) because the default for torch.nn.Parameter is requires_grad = True
            self.fc_r2y.bias.requires_grad = False# y = Wyh @ h + by
            self.n_parameters = n_recurrent**2 + n_recurrent*n_input + n_recurrent + n_output*n_recurrent# number of learned parameters in model
        #------------------------------
        # initialize input(Wrx), recurrent(Wrr), output(Wyr) weights 
        if Wrx is not None:
            self.fc_x2r.weight = torch.nn.Parameter(Wrx)# Wrx @ x + br
        if Wrr is not None:
            self.fc_r2r.weight = torch.nn.Parameter(Wrr)# Wrr @ r
        if Wyr is not None:
            self.fc_r2y.weight = torch.nn.Parameter(Wyr)# y = Wyr @ r + by
        #------------------------------
        if LEARN_OUTPUTWEIGHT==False:# this must go after the line self.fc_r2y.weight = torch.nn.Parameter(Wyr) because the default for torch.nn.Parameter is requires_grad = True, if LEARN_OUTPUTWEIGHT = False then Wyr does not change during gradient descent learning
            self.fc_r2y.weight.requires_grad = False# y = Wyr @ r + br
            self.n_parameters = self.n_parameters - n_output*n_recurrent# number of learned parameters in model
        #------------------------------
        # set the activation function for r 
        # pytorch seems to have difficulty saving the model architecture when using lambda functions
        # https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/3
        #self.activation_function = lambda x: f(x, activation_function)
        self.activation_function = activation_function
        #------------------------------
        # set the initial state r0
        if r0 is None:
            self.r0 = torch.nn.Parameter(torch.zeros(n_recurrent), requires_grad=False)# (n_recurrent,) tensor
        else:
            self.r0 = torch.nn.Parameter(r0, requires_grad=False)# (n_recurrent,) tensor
        if LEARN_r0:
            #self.ah0 = self.ah0.requires_grad=True# learn initial value for h, https://discuss.pytorch.org/t/learn-initial-hidden-state-h0-for-rnn/10013/6  https://discuss.pytorch.org/t/solved-train-initial-hidden-state-of-rnns/2589/8
            self.r0 = torch.nn.Parameter(self.r0, requires_grad=True)# (n_recurrent,) tensor
            self.n_parameters = self.n_parameters + n_recurrent# number of learned parameters in model
        #------------------------------
        
        
    # output y for all n_T timesteps   
    def forward(self, model_input_forwardpass):# nn.Linear expects inputs of size (*, n_input) where * means any number of dimensions including none
        input = model_input_forwardpass['input']
        brneverlearn = model_input_forwardpass['activity_noise']
        if len(input.shape)==2:# if input has size (n_T, n_input) because there is only a single trial then add a singleton dimension
            input = input[None,:,:]# (n_trials, n_T, n_input)
            brneverlearn = brneverlearn[None,:,:]# (n_trials, n_T, n_recurrent)
        dt = self.dt
        Tau = self.Tau
        #n_trials, n_T, n_input = input.size()# METHOD 1
        n_trials, n_T, n_input = input.shape# METHOD 2
        #n_recurrent = self.fc_r2y.weight.size(1)# y = Wyr @ r + by, METHOD 1
        #n_recurrent = self.fc_r2y.weight.shape[1]# y = Wyr @ r + by, METHOD 2
        r = self.r0.repeat(n_trials, 1)# (n_trials, n_recurrent) tensor, all trials should have the same initial value for r, not different values for each trial
        rstore = []# (n_trials, n_T, n_recurrent)
        for t in range(n_T):
            r = r + (dt/Tau) * (-r + compute_activation_function( self.fc_r2r(r) + self.fc_x2r(input[:, t]), self.activation_function)  + brneverlearn[:,t,:])# brneverlearn has shape (n_trials, n_T, n_recurrent) 
            rstore.append(r)# rstore += [r]
        rstore = torch.stack(rstore,dim=1)# (n_trials, n_T, n_recurrent), each appended r is stored in rstore[:,i,:], nn.Linear expects inputs of size (*, n_recurrent) where * means any number of dimensions including none 
        #return self.fc_r2y(rstore), rstore# (n_trials, n_T, n_output/n_recurrent) tensor, y = Wyr @ r + by
        model_output_forwardpass = {'output':self.fc_r2y(rstore), 'activity':rstore}# (n_trials, n_T, n_output/n_recurrent) tensor, y = Wyr @ r + by
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







    

    