'''
Make correction. Sample should come before forward(), and not after it. since we are sending it inputs[0]
hprev and cprev have to be decided accordingly.
'''


from layers import forward, backward


import numpy as np


# First we create the data variable.

# data = open('HarryPotter.txt', 'r').read()
# data = data.split()




import cv2

img = cv2.imread("Bday Treat.jpg",cv2.IMREAD_GRAYSCALE)



# print(img.shape)

# cv2.imshow("Original",img)

r = 70 / img.shape[1]
dim = (70, int(img.shape[0] * r))
 
# perform the actual resizing of the image and show it
resizeds = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

data = list(resizeds.reshape((resizeds.shape[0]*resizeds.shape[1],    )))

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print( 'data has {0} characters, {1} unique.'.format(data_size, vocab_size) )
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1


# The +1 is for the bias in both WLSTM and WHY
WLSTM = np.random.randn(1 + vocab_size + hidden_size, 4*hidden_size)
WHY = np.random.randn(1 + hidden_size, vocab_size)
# IFOG = np.zeros( (data_size, 4*hidden_size) )
# IFOGf = np.zeros_like(IFOG)

X = np.zeros((data_size, vocab_size))

# The +1 with hidden_size is for the biases
Hout = np.zeros((data_size, hidden_size))
# Hout[:,0] = np.ones(data_size)

Y = np.zeros((data_size, vocab_size))
P = np.zeros((data_size, vocab_size))

C = np.zeros((data_size, hidden_size))


dY = np.zeros_like(Y)
dHout = np.zeros_like(Hout)
dhout = np.zeros((data_size, hidden_size+1))
dCt = np.zeros((data_size, hidden_size))
dC = np.zeros((data_size, hidden_size))
dWHY = np.zeros_like(WHY)


mWLSTM, mWHY = np.zeros_like(WLSTM), np.zeros_like(WHY)
# Might not need some of the below. 
# e=0.3
txtf = None
n, p = 0, 0
'''
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
'''
loss=0
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
e=0.03

do_from_start = True




def sample(h, c, seed_ix, n):
#{
    x = np.zeros(vocab_size,)
    x[seed_ix]=1
    ixes=[]

    xphpb = WLSTM.shape[0] # x plus h plus bias, lol
    H_in = np.zeros((n, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
    h_out = np.zeros((n, hidden_size)) # hidden representation of the LSTM (gated cell content)
    IFOG_ = np.zeros((n, int(hidden_size * 4))) # input, forget, output, gate (IFOG)
    IFOGf_ = np.zeros((int(n), int(hidden_size * 4))) # after nonlinearity
    C_ = np.zeros((n, hidden_size )) # cell content
    Ct_ = np.zeros((n, hidden_size )) # tanh of cell content

    for t in range(n):
    #{ 
        # concat [x,h] as input to the LSTM

        
        if t > 0:
        #{
            prevh = h_out[t-1]
        #}

        else:
        #{
            prevh = h
        #}
        
        H_in[t,0] = 1 # bias
        H_in[t, 1:vocab_size+1] = x
        H_in[t, vocab_size+1:] = prevh
        # compute all gate activations. dots: (most work is this line)
        IFOG_[t] = np.dot(H_in[t].reshape((1,xphpb)),WLSTM).reshape(4*hidden_size,)
        # non-linearities
        IFOGf_[t,:int(3*hidden_size)] = 1.0/(1.0+np.exp(-IFOG_[t,:int(3*hidden_size)])) # sigmoids; these are the gates
        IFOGf_[t,int(3*hidden_size):] = np.tanh(IFOG_[t,int(3*hidden_size):]) # tanh
        # compute the cell activation
        prevc = C_[t-1] if t > 0 else c
        
        C_[t] = IFOGf_[t,:hidden_size] * IFOGf_[t,3*hidden_size:] + IFOGf_[t,hidden_size:2*hidden_size] * prevc
        Ct_[t] = np.tanh(C_[t])

        # The dimension is (hidden_size, )
        h_out[t] = IFOGf_[t,2*hidden_size:3*hidden_size] * Ct_[t]
        ho = np.copy( np.hstack( (np.ones((n,1)), h_out)  ) )
        y = np.dot( WHY.T , ho[t].reshape((hidden_size+1, 1 )) ).reshape(vocab_size,)
        p = np.exp(y) /  np.sum( np.exp(y) )
        ix = list(p).index(max(list(p)))

        # ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros(vocab_size,)
        
        x[ix] = 1
        ixes.append(ix)
    #}
    return(ixes)

    
#}













# The approach of taking subsequences to learn is what makes the code a little complicated here.
flag=False
while(n<19300):
#{
    
    if(do_from_start == True):
    #{
        # We'll create them as arrays and not matrices since we will be assigning them to rows of a particular matrix.
        hprev = np.zeros(hidden_size,)
        cprev = np.zeros(hidden_size,)

        # Go from start of the data
        p=0
        seq_length = 25
        do_from_start = False
    #}

    if(p+seq_length+1 >= data_size):
    #{
        '''

        If the next pass will get us to the end of the data, then do that pass. After that, start backpropagation.
        How do we jump to that part? We can do that by writing a while loop in this that does the backpropagation for the
        entire data in sub_sequences.
        Once we have obtained all the derivatives we need to update our parameters, we will update our parameters.
        Then, we will end that while loop. Outside it, we will set do_from_start to True, and continue with the entire outer loop again.
        
        '''
        
        # I will first write code to forward pass through the final subsequence
        if(p+seq_length+1==len(data)):
        #{
            '''
            if(flag):
            #{
                cprev=dc0
                hprev = dh0
            #}
            '''
            inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
            cprev, hprev, cache_from_forward = forward( X[p:p+seq_length, :], inputs, targets, WLSTM, Hout[p:p+seq_length, :], cprev, hprev )
            current_subseq_IFOGf = cache_from_forward['IFOGf']
            current_subseq_Ct = cache_from_forward['Ct']
            hout = np.copy( np.hstack( (np.ones((data_size,1)), Hout)  ))
            count=0
            for time_step in range(p, p +seq_length):
            #{
                loss=0
                Y[time_step] = np.dot( WHY.T , hout[time_step].reshape((hidden_size+1, 1 )) ).reshape(vocab_size,)
                P[time_step] = np.exp(Y[time_step]) /  np.sum( np.exp(Y[time_step]) )
                loss += -np.log( P[time_step][targets[count]] )
                dY[time_step] = np.copy(P[time_step])
                dY[time_step][targets[count]] -= 1
                dhout[time_step] = np.dot( WHY, dY[time_step].reshape((vocab_size,1))).reshape(hidden_size+1,)
                dHout[time_step] = dhout[time_step,1:]

                dCt[time_step] = dHout[time_step]*current_subseq_IFOGf[count, 2*hidden_size: 3*hidden_size]
                dC[time_step] = (1-current_subseq_Ct[count]**2) * dCt[time_step]
                dC[time_step-1] = current_subseq_IFOGf[count,hidden_size:2*hidden_size] * dC[time_step]
                dWHY = np.dot(  hout[time_step].reshape((hidden_size+1, 1 )), dY[time_step].reshape((1,vocab_size))  )
                count+=1
                if(count==len(targets)):
                    count-=1
                            
                
                
                
                
            #}
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            
        

            
        #}
        elif(p+seq_length+1>len(data)):
        #{
            
            seq_length = ( len(data) ) - p
            inputs = [char_to_ix[ch] for ch in data[p:p+seq_length-1]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
            cprev, hprev, cache_from_forward = forward( X[p:p+seq_length, :], inputs, targets, WLSTM, Hout[p:p+seq_length, :], cprev, hprev )
            current_subseq_IFOGf = cache_from_forward['IFOGf']
            current_subseq_Ct = cache_from_forward['Ct']
            hout = np.copy( np.hstack( (np.ones((data_size,1)), Hout)  ) )
            count=0
            for time_step in range(p, p +seq_length):   
            #{
                loss=0
                Y[time_step] = np.dot( WHY.T , hout[time_step].reshape((hidden_size+1, 1 )) ).reshape(vocab_size,)
                P[time_step] = np.exp(Y[time_step]) /  np.sum( np.exp(Y[time_step]) )
                loss += -np.log( P[time_step][targets[count]] )
                dY[time_step] = np.copy(P[time_step])
                dY[time_step][targets[count]] -= 1
                dhout[time_step] = np.dot( WHY, dY[time_step].reshape((vocab_size,1))).reshape(hidden_size+1,)
                dHout[time_step] = dhout[time_step,1:]

                dCt[time_step] = dHout[time_step]*current_subseq_IFOGf[count, 2*hidden_size: 3*hidden_size]
                dC[time_step] = (1-current_subseq_Ct[count]**2) * dCt[time_step]
                dC[time_step-1] = current_subseq_IFOGf[count,hidden_size:2*hidden_size] * dC[time_step]
                count+=1
                if(count==len(targets)):
                    count-=1
                
            #}
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            
        #}
        '''
        # Now, I will write the while loop for backward propagation in subsequences.
        # For that, p will have to be initialized to the index, to which if we add seq_length, then it will equal data_length, i.e. the data_size-1
        # p = data_size - seq_length - 1
        
        # while(p>=0):
        #{
                   
            inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
            
            


            
            




            # p-=seq_length
        #}
        '''
        _, dWLSTM, dc0, dh0 = backward(inputs, dHout[p:p+seq_length, :], cache_from_forward, dC[time_step], dHout[time_step])
        flag=True
                            
        for param, dparam, mem in zip([WLSTM, WHY], [dWLSTM, dWHY], [mWLSTM, mWHY]):
        #{
            mem+=dparam*dparam
            param+= (-learning_rate*dparam/(np.sqrt(mem+1e-8)))
        #}
        if n % 100 == 0:
        #{
            sample_ix = sample(hprev,cprev, inputs[0], data_size)
            print( 'iter {0}, loss: {1}'.format(n, smooth_loss)) # print progress
            # txt = ''.join(ix_to_char[ix]+' ' for ix in sample_ix)
            txt = np.array([ix_to_char[ix] for ix in sample_ix])
            print(txt)
        #}
        
        do_from_start = True
        continue                
    #}

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    cprev, hprev, cache_from_forward = forward( X[p:p+seq_length, :], inputs, targets, WLSTM, Hout[p:p+seq_length, :], cprev, hprev )

    current_subseq_IFOGf = cache_from_forward['IFOGf']
    current_subseq_Ct = cache_from_forward['Ct']
    # current_subseq_C = cache_from_forward['C']

    
    hout = np.copy( np.hstack( (np.ones((data_size,1)), Hout)  ) )

    count=0
    for time_step in range(p, p +seq_length):
    #{
        loss=0
        Y[time_step] = np.dot( WHY.T , hout[time_step].reshape((hidden_size+1, 1 )) ).reshape(vocab_size,)
        P[time_step] = np.exp(Y[time_step]) /  np.sum( np.exp(Y[time_step]) )
        loss += -np.log( P[time_step][targets[count]] )

        dY[time_step] = np.copy(P[time_step])
        dY[time_step][targets[count]] -= 1

        # The dHout calculated here is only a small part of dHout. We will be accumulating them during backward.s
        # # There's something wrong with the dimensions here. Gonna have to check that.
        dhout[time_step] = np.dot( WHY, dY[time_step].reshape((vocab_size,1))).reshape(hidden_size+1,)
        dHout[time_step] = dhout[time_step,1:] 
        
        # Now I have to calculate dC[time_step] too.
        # For that, I first need to calculate dCt too. For calculating dCt, I will need Hout[time_step]

        # In the backward pass, we write code so that accumulation of gradients is there. Here, we just have to
        # calculate dC[time_step] only due to the Hout[time_step] and not due to the Houts of later time_steps.
        dCt[time_step] = dHout[time_step]*current_subseq_IFOGf[count, 2*hidden_size: 3*hidden_size]
        dC[time_step] = (1-current_subseq_Ct[count]**2) * dCt[time_step] 
        dC[time_step-1] = current_subseq_IFOGf[count,hidden_size:2*hidden_size] * dC[time_step]

        # I should also calculate dWHY.
        dWHY = np.dot(  hout[time_step].reshape((hidden_size+1, 1 )), dY[time_step].reshape((1,vocab_size))  )
        for i in [dWHY]:
            np.clip(i,-5,5,out=i)
            
        count+=1
        if(count==len(targets)):
            count-=1

        
    #}


    
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # The backward pass for each subsequence should happen right after the forward pass for that subsequence.
    # That way, the cache variables can be made use of.

    _, dWLSTM, dc0, dh0 = backward(inputs, dHout[p:p+seq_length, :], cache_from_forward, dC[time_step], dHout[time_step])

    for param, dparam, mem in zip([WLSTM, WHY], [dWLSTM, dWHY], [mWLSTM, mWHY]):
    #{
        mem+=dparam*dparam
        param += (-learning_rate*dparam/(np.sqrt(mem+1e-8)))
    #}

    if n % 100 == 0:
    #{
        sample_ix = sample(hprev,cprev, inputs[0], data_size)
        print( 'iter {0}, loss: {1}'.format(n, smooth_loss)) # print progress
        # txt = ''.join(ix_to_char[ix]+' ' for ix in sample_ix)
        txt = np.array([ix_to_char[ix] for ix in sample_ix])
        print(txt)
    #}

    # I don't see any use for dc0 and dh0. Since, I am backpropagating along the time_steps for each subsequence, there's no point
    # in remembering the last time step's dC and dHout. 

     
    # subsequence counter
    p+=seq_length
    # iteration counter
    n+=1
    # Iteration counter keeping track of total forward and backward passes.
    

#}
# the_main()
def chat():
#{
    choice = 'No'
    if(choice == 'No'):
    #{
        
        # hprev = np.random.randn(hidden_size,)
        # cprev = np.random.randn(hidden_size,)
        # question = input('Shubham: ')
        # s = question.split()
        x = np.zeros(vocab_size,)
        inputs = char_to_ix[data[-1]]
        '''
        x[inputs]=1
        # I have to look for the Hout[t] and C[t] that come just before the x[inputs]'s timestep
        for i in range(data_size):
        #{
            if(not False in list(X[i] == x)):
                if(i==0):
                    hprev = np.zeros(hidden_size,)
                    cprev = np.zeros(hidden_size,)
                
                hprev = Hout[i-1]
                cprev = C[i-1]
                break
        #}
        '''
        # hprev = Hout[data_size-1]
        # cprev = C[data_size-1]
        hprev = np.zeros(hidden_size,)
        cprev = np.zeros(hidden_size,)
        
        sample_ix = sample(hprev,cprev, data[0], resizeds.shape[0]*resizeds.shape[1])
        # txt = ''.join(ix_to_char[ix]+' ' for ix in sample_ix)
        txt = np.array([ix_to_char[ix] for ix in sample_ix])
        # print('\n Bonnie Says: {0}\n'.format(txt))
        
        txt = txt.reshape((resizeds.shape[0],resizeds.shape[1]))
        cv2.imshow('Constructed',txt)
    
        # choice = input('\nYou done?: ')
    
    



    
    #}
#}  




chat()










