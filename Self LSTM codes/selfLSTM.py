"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
import code

class LSTM:
#{
  
  @staticmethod
  def init(input_size, hidden_size, fancy_forget_bias_init = 3):
  #{
    """ 
    Initialize parameters of the LSTM (both weights and biases in one matrix) 
    One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
    """
    # +1 for the biases, which will be the first row of WLSTM

    # ================== Why 4*Hidden Size?

    WHY = np.random.randn(hidden_size, input_size)
    
    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
    WLSTM[0,:] = 0 # initialize biases to zero

    WHY[0,:] = 0

    if fancy_forget_bias_init != 0:
    # {
      # forget gates get little bit negative bias initially to encourage them to be turned off
      # remember that due to Xavier initialization above, the raw output activations from gates before
      # nonlinearity are zero mean and on order of standard deviation ~1
      WLSTM[0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    # }

    

    
    return WLSTM,WHY
  # }
  
  @staticmethod
  def lossFun(inputs, targets, WHY, WLSTM, n, input_size, c0 = None, h0 = None):
  #{
    """
    X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
    """
    X,Y,P = {},{},{}

    # ------------- NOPES! X is going to be a dictionary --------------

    # ------------------------------------------------- inputs parameter contains the data! targets contains what is supposed to be predicted --------------
    # =============== n,b,input_size = X.shape

    # I am going to make X a 2-D matrix, instead of 3. I am going to get rid of the batches ! (length of sequence, No. of unique characters (or elements) )


    # ------------------------------ n represents vocab size and input_size represents no. of characters.
    # ------------- n, input_size = X.shape

    # ----------- H[-1] = np.copy(hprev)
    loss=0

    
    d = int(WLSTM.shape[1]/4 )# hidden size
    if c0 is None:
      # c0 ======= np.zeros((b,d))
      c0 = np.zeros((d,1))
    
    if h0 is None:
      # ======= h0 = np.zeros((d,1))
      h0 = np.zeros((d,1))
    
    
    # Perform the LSTM forward pass with X as the input
         
    for t in range(n):
    #{
      # concat [x,h] as input to the LSTM
      # xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
      # xs[t][inputs[t]] = 1

      #   INPUT_SIZE IS VOCAB_SIZE OR BASICALLY NUMBER OF UNIQUE ELEMENTS IN THE DATA!

      X[t] = np.zeros((input_size,1))
      X[t][ inputs[t] ] = 1


      
      # -------------------- prevh = Hout[t-1] if t > 0 else h0

      prevh = Hout[t-1] if t>0 else h0


      # ====================== Hin[t,:,0] = 1 # bias

      Hin[t,0] = 1
      
      
      # ====================== Hin[t,:,1:input_size+1] = X[t]
      Hin[t,1:input_size+1] = X[t].reshape(X[t].shape[0])
      
      # ==================== Hin[t,:,input_size+1:] = prevh

      Hin[t, input_size+1:] = prevh.reshape(prevh.shape[0])

      
      # compute all gate activations. dots: (most work is this line)


      # -------------------------------- Shape of IFOG[t] -> (1+xphb,)
      IFOG[t] = Hin[t].reshape((1,Hin[t].shape[0])).dot(WLSTM) 



      # non-linearities
      # ============================ IFOGf[t,:,:int(3*d)] = 1.0/(1.0+np.exp(-IFOG[t,:,:int(3*d)])) # sigmoids; these are the gates

      IFOGf[t,:3*d] = 1.0/(1.0 + np.exp(-(IFOG[t,:3*d])))
      
      # =============================== IFOGf[t,:,int(3*d):] = np.tanh(IFOG[t,:,int(3*d):]) # tanh
      
      IFOGf[t, 3*d:] = np.tanh(  IFOG[t , 3*d:]   )

      
      # compute the cell activation
      prevc = C[t-1] if t > 0 else c0

      
      # ====================================== C[t] = IFOGf[t,:,:int(d)] * IFOGf[t,:,int(3*d):] + IFOGf[t,:,int(d):int(2*d)] * prevc

      # ------------------ shape of C[t] is (d,)
      C[t] = IFOGf[t, :d]*IFOGf[t,3*d:]*prevc.reshape(prevc.shape[0]) 
      
      Ct[t] = np.tanh(C[t])
      # ======================== Hout[t] = IFOGf[t,:,int(2*d):int(3*d)] * Ct[t]

      # ------------------------- Shape of Hout[t] is (d,)
      Hout[t] = IFOGf[t, 2*d:3*d]*Ct[t]


      # ---------------------------- Now, I need to write the code for calculating the Y[t] part---------------------
      '''
      ----------

      I have to multiply (or dot product) Hout[t] with the weights matrix to get y[t]

      Y[t] is supposed to be (input_size,) shape! - So the WHY weight matrix should be (d, input_size)
      ----------
      '''
      # ------------ Hout[t] is (d,)
      Y[t] = np.dot(Hout[t].reshape((1,Hout[t].shape[0])),WHY)
      P[t] = np.exp(Y[t]) / np.sum(np.exp(Y[t]))
      loss += -np.log(P[t][0,targets[t]])
      # --------- ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
      # ---------- ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
      # ----------- loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    #}
    for_returning = t

      

    
    '''
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
    ''' 
    # -------------------------- return C[t], as well so we can continue LSTM with prev state init if needed
    # ------------------------------ return Hout, C[t], Hout[t], cache, loss
    
    # ============ if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
    # ============ if dhn is not None: dHout[n-1] += dhn.copy()
    for t in reversed(range(n)):
    #{
      dY=np.copy(P[t])

      dY[0,targets[t]] = dY[0,targets[t]] - 1

      # -------------------------- I am going to have to reshape dY and Hout[t] from (input_size,) and (d,) to (1,input_size) and (d,1)
      dY = dY.reshape((1,dY.shape[1]))
      Hout[t] = Hout[t].reshape(Hout[t].shape[0])


      global dWHY
      dWHY = dWHY + np.dot(Hout[t].reshape((Hout[t].shape[0],1)),dY)
        


      dHout[t] = np.dot(WHY ,dY.T).reshape(   np.dot(WHY ,dY.T).shape[0]   )
 
      tanhCt = Ct[t]
      # --------------------- dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]
      #--------------------------- dIFOGf[t,:,2*d:3*d] = tanhCt * dy

      dIFOGf[t, 2*d:3*d] = tanhCt * dHout[t]
      # backprop tanh non-linearity first then continue backprop

      # ---------------------- dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])
      dC[t] += (1-tanhCt**2) * (IFOGf[t,2*d:3*d] * dHout[t])
      
      
 
      if t > 0:
      #{
        # --------------- dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
        dIFOGf[t,d:2*d] = C[t-1] * dC[t]
      
        
        # -----------------dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
        dC[t-1] += IFOGf[t,d:2*d] * dC[t]
      #}
      else:
      #{
        # -------------- dIFOGf[t,:,d:2*d] = c0 * dC[t]
        dIFOGf[t,d:2*d] = c0.reshape(c0.shape[0]) * dC[t].reshape(dC[t].shape[0])
        
        # ---------- dc0 = IFOGf[t,:,d:2*d] * dC[t]
        dc0 = IFOGf[t,d:2*d] * dC[t]
      #}
      
      # ------------- dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
      dIFOGf[t,:d] = IFOGf[t,3*d:] * dC[t]
      # ------------- dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]
      dIFOGf[t,3*d:] = IFOGf[t,:d] * dC[t]
      
      # backprop activation functions
      
      # ----------- dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
      dIFOG[t, 3*d:] = (1 - IFOGf[t,3*d:] ** 2) * dIFOGf[t,3*d:]
      
      # ------------- y = IFOGf[t,:,:3*d]
      y = IFOGf[t,:3*d]
      # ------------- dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]
      dIFOG[t,:3*d] = (y*(1.0-y)) * dIFOGf[t, :3*d]
 
      # backprop matrix multiply
      global dWLSTM

      dWLSTM = dWLSTM + np.dot(Hin[t].reshape((Hin[t].shape[0],1)), dIFOG[t].reshape(  (1,dIFOG[t].shape[0])   )    )



        
      dHin[t] = dIFOG[t].reshape((1,dIFOG[t].shape[0])).dot(WLSTM.transpose())
 
      # backprop the identity transforms into Hin
      # =============== dX[t] = dHin[t,:,1:input_size+1]
      dX[t] = dHin[t,1:input_size+1]
      if t > 0:
        # ============ dHout[t-1,:] += dHin[t,:,input_size+1:]
        dHout[t-1,:] += dHin[t, input_size+1:]
      else:
        
        #=============dh0 += dHin[t,:,input_size+1:]
        global dh0
        dh0 = dh0 + dHin[t,input_size+1:].reshape((dHin[t,input_size+1:].shape[0],1))
        
    #}
 
    # --------- return dX, dWLSTM, dc0, dh0, loss
    return(dWLSTM,dWHY, Hout[for_returning-1], C[for_returning-1], loss)
  

  
    
  #}
#}









data = open('potter.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)





no, p = 0, 0
seq_length = 25
hidden_size = 100



print( 'data has {0} characters, {1} unique.'.format(data_size, vocab_size) )


WLSTM,WHY = LSTM.init(vocab_size,100)
mWLSTM, mWHY = np.zeros_like(WLSTM), np.zeros_like(WHY) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
cprev = np.zeros((hidden_size,1))
hprev = np.zeros((hidden_size,1))

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
learning_rate = 1e-1
d = int(WLSTM.shape[1]/4 )# hidden size

inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]

xphpb = WLSTM.shape[0] # x plus h plus bias, lol
    # ========= Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
Hin = np.zeros((len(inputs), xphpb))
    
    # ============ Hout = np.zeros((n, b, int(d))) # hidden representation of the LSTM (gated cell content)
Hout = np.zeros((len(inputs),d))
    
    # ========== IFOG = np.zeros((n, b, int(d * 4))) # input, forget, output, gate (IFOG)

IFOG = np.zeros((len(inputs),4*d))
    # ============ IFOGf = np.zeros((int(n), int(b), int(d * 4))) # after nonlinearity
IFOGf = np.zeros((len(inputs),4*d))
    
    # ======== C = np.zeros((n, b, int(d) )) # cell content
C = np.zeros((len(inputs),d))
    
    # ============= Ct = np.zeros((n, b, int(d) )) # tanh of cell content
Ct = np.zeros((len(inputs),d))





dIFOG = np.zeros(IFOG.shape)
dIFOGf = np.zeros(IFOGf.shape)
global dWLSTM
dWLSTM = np.zeros(WLSTM.shape)
global dWHY
dWHY = np.zeros(WHY.shape)
dHin = np.zeros(Hin.shape)
dC = np.zeros(C.shape)
dX = np.zeros((len(inputs),vocab_size))
global dh0
dh0 = np.zeros((d,1))
dc0 = np.zeros((d,1))
dHout = np.zeros((len(inputs),d)) # make a copy so we don't have any funny side effects












def sample(h, c, seed_ix, n):
#{
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  Hin = np.zeros((n, vocab_size+hidden_size+1))
  Hout = np.zeros((n,hidden_size))
  IFOG = np.zeros((n,4*hidden_size))
  IFOGf = np.zeros((n,4*hidden_size))
  C = np.zeros((n,hidden_size))
  Ct = np.zeros((n,hidden_size))





  
  x = np.zeros((vocab_size,))
  x[seed_ix] = 1
  ixes = []
  
  for t in range(n):
  #{
    Hin[t,0] = 1
    Hin[t,1:vocab_size+1] = x.reshape(x.shape[0])
    Hin[t, vocab_size+1:] = h.reshape(h.shape[0])
    IFOG[t] = Hin[t].dot(WLSTM)
    IFOGf[t,:3*hidden_size] = 1.0/(1.0 + np.exp(-(IFOG[t,:3*hidden_size])))
    IFOGf[t, 3*hidden_size:] = np.tanh(  IFOG[t , 3*hidden_size:]   )

    C[t] = IFOGf[t, :hidden_size]*IFOGf[t,3*hidden_size:]*c.reshape(c.shape[0])
    Ct[t] = np.tanh(C[t])
    Hout[t] = IFOGf[t, 2*hidden_size:3*hidden_size]*Ct[t]
    Y = np.dot(Hout[t],WHY)

    P = np.exp(Y) / np.sum(np.exp(Y))
    



    
    
    '''
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    '''
    ix = list(P).index(max(list(P)))
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  #}
  return ixes
#}

















while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or no == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data

  # -------------------------------------------------------------- The next 2 lines are highly important  
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if no % 100 == 0:
    sample_ix = sample(hprev, cprev, inputs[0], len(data))
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print( '----\n {0} \n----'.format(txt ))

  # forward seq_length characters through the net and fetch gradient

  #WLSTM,WHY = LSTM.init(vocab_size,100)
  
  dWLSTM, dWHY, hprev, cprev, loss = LSTM.lossFun(inputs, targets, WHY, WLSTM, len(inputs), vocab_size, c0=cprev, h0=hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if no % 100 == 0: print( 'iter {0}, loss: {1}'.format(no, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([WLSTM, WHY], 
                                [dWLSTM, dWHY], 
                                [mWLSTM, mWHY]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  no += 1 # iteration counter
  
  
  
