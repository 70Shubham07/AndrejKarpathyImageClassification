'''
def init(input_size, hidden_size, fancy_forget_bias_init = 3):
#{   
    """ 
    Initialize parameters of the LSTM (both weights and biases in one matrix) 
    One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
    """
    # +1 for the biases, which will be the first row of WLSTM

    # ================== Why 4*Hidden Size?

    
    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
    WLSTM[0,:] = 0 # initialize biases to zero

    if fancy_forget_bias_init != 0:
    #{
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        WLSTM[0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    #}
    
    return WLSTM
#}
'''  
import numpy as np
def forward(X, inputs, targets, WLSTM, Hout, c0 = None, h0 = None):
#{  
    """
    X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
    """

    loss = 0
    
    n,input_size = X.shape
    d = int(WLSTM.shape[1]/4 )# hidden size
    if c0 is None:
      c0 = np.zeros(d,)
    if h0 is None:
      h0 = np.zeros(d,)
    
    # Perform the LSTM forward pass with X as the input
    xphpb = WLSTM.shape[0] # x plus h plus bias, lol
    Hin = np.zeros((n, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
    # Hout = np.zeros((n, b, int(d))) # hidden representation of the LSTM (gated cell content)
    IFOG = np.zeros((n, int(d * 4))) # input, forget, output, gate (IFOG)
    IFOGf = np.zeros((int(n), int(d * 4))) # after nonlinearity
    C = np.zeros((n, int(d) )) # cell content
    Ct = np.zeros((n, int(d) )) # tanh of cell content

    for t in range(len(inputs)):
    #{ 
        # concat [x,h] as input to the LSTM

        X[t][inputs[t]] = 1
        if t > 0:
        #{
            prevh = Hout[t-1]
        #}

        else:
        #{
            prevh = h0
        #}
        
        Hin[t,0] = 1 # bias
        Hin[t, 1:input_size+1] = X[t]
        Hin[t, input_size+1:] = prevh
        # compute all gate activations. dots: (most work is this line)
        IFOG[t] = np.dot(Hin[t].reshape((1,xphpb)),WLSTM).reshape(4*int(d),)
        # non-linearities
        IFOGf[t,:int(3*d)] = 1.0/(1.0+np.exp(-IFOG[t,:int(3*d)])) # sigmoids; these are the gates
        IFOGf[t,int(3*d):] = np.tanh(IFOG[t,int(3*d):]) # tanh
        # compute the cell activation
        prevc = C[t-1] if t > 0 else c0
        
        C[t] = IFOGf[t,:int(d)] * IFOGf[t,int(3*d):] + IFOGf[t,int(d):int(2*d)] * prevc
        Ct[t] = np.tanh(C[t])

        # The dimension is (hidden_size, )
        Hout[t] = IFOGf[t,int(2*d):int(3*d)] * Ct[t]

        

        

    #}

    cache = {}
    cache['WLSTM'] = WLSTM
    cache['Hout'] = Hout
    cache['IFOGf'] = IFOGf
    cache['IFOG'] = IFOG
    cache['C'] = C
    cache['Ct'] = Ct
    cache['Hin'] = Hin
    cache['c0'] = c0
    cache['h0'] = h0

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return C[t], Hout[t], cache
#}
  

def backward(inputs, dHout_in, cache, dcn = None, dhn = None):
#{  

    WLSTM = cache['WLSTM']
    Hout = cache['Hout']
    IFOGf = cache['IFOGf']
    IFOG = cache['IFOG']
    C = cache['C']
    Ct = cache['Ct']
    Hin = cache['Hin']
    c0 = cache['c0']
    h0 = cache['h0']
    n,d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1 # -1 due to bias
 
    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n,input_size))
    dh0 = np.zeros(d,)
    dc0 = np.zeros(d,)
    dHout = dHout_in.copy() # make a copy so we don't have any funny side effects
    if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
    if dhn is not None: dHout[n-1] += dhn.copy()
    for t in reversed(range(n)):
    #{ 
 
        tanhCt = Ct[t]
        dIFOGf[t,2*d:3*d] = tanhCt * dHout[t]
        # backprop tanh non-linearity first then continue backprop
        dC[t] += (1-tanhCt**2) * (IFOGf[t,2*d:3*d] * dHout[t])
 
        if t > 0:
        #{
            dIFOGf[t,d:2*d] = C[t-1] * dC[t]
            dC[t-1] += IFOGf[t,d:2*d] * dC[t]
        #}
        else:
        #{
            dIFOGf[t,d:2*d] = c0 * dC[t]
            dc0 = IFOGf[t,d:2*d] * dC[t]
        #}
      
        dIFOGf[t,:d] = IFOGf[t,3*d:] * dC[t]
        dIFOGf[t,3*d:] = IFOGf[t,:d] * dC[t]
      
        # backprop activation functions
        dIFOG[t,3*d:] = (1 - IFOGf[t,3*d:] ** 2) * dIFOGf[t,3*d:]
        y = IFOGf[t,:3*d]
        dIFOG[t,:3*d] = (y*(1.0-y)) * dIFOGf[t,:3*d]
 
        # backprop matrix multiply
        dWLSTM += np.dot( Hin[t].reshape((WLSTM.shape[0],1)), dIFOG[t].reshape((1,int(4*d))))
        dHin[t] = dIFOG[t].dot(WLSTM.transpose())
 
        # backprop the identity transforms into Hin
        dX[t] = dHin[t,1:input_size+1]
        if t > 0:
            dHout[t-1,:] += dHin[t,input_size+1:]
        else:
            dh0 += dHin[t,input_size+1:]
            

    #}
    for i in [dWLSTM]:
        np.clip(i, -5,5,out=i)
            
 
    return dX, dWLSTM, dc0, dh0
#}

