import numpy as np
from random import shuffle
'''
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)

    ## Computing de/dw grad.
    
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW
'''

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero    ------> (n, c)
  num_train = X.shape[0]        # ---> m
  num_classes = W.shape[1] # ----> c
  num_features = X.shape[1] # -----> n+1 (+1 for biases which were included earlier in the code.)

  
  scores = np.dot(X,W)   #Each row contains scores for classes, while the columns are classes    ---->   (m,n+1)*(n+1,c)   =  ( m,c  )
  
  #Next, we have to sutract the original class scores of each row, from their respective rows. We can do that with broadcsting.
   
  col_of_original_scores =  ( scores[range(num_train) , list( y )] ).reshape(-1,1)    # ----> (m,1) the (-1,1) turned it into a vector.
  the_scores_before_taking_max = scores - (col_of_original_scores  - np.ones( (num_train,1)  ))  # ---> (m,c) . From each element of scores, corresponding row's element in col_of_original_scores
                                                                                                 # will get subtracted.
                                                                    
  the_margins_matrix = np.maximum( np.zeros( (num_train,num_classes) ) ,  the_scores_before_taking_max  )   # -----> (m,c)
  
  loss_without_reg = np.sum(the_margins_matrix)/ (num_train)    # ------ Loss averaged over number of training data.

  loss_with_reg = (1/2)*reg*(np.sum(W*W))

  ''' Next, calculating the gradients matrix '''
  the_first_ones_matrix = np.zeros( (num_train, num_classes)  )

  the_first_ones_matrix[ the_margins_matrix>0 ] =  1    #By boolean masking, I have gotten rid of all the weights that won't be contributing to loss.

  '''The logic behind next step :-
     the dW matrix's each element will be derived by differentiating "loss_without_reg" w.r.t. the respective element of W matrix.
     Since, in loss matrix, each element is of the form weight*x. Thus, we know that the gradients will consist only of "x" sums.
     The question is, which 'w' appears how many times in loss_without_reg matrix, with what co-efficient , and in multiplication with which 'x'. Thus, gradient
     of loss, w.r.t. that 'w', will be the sum of the 'x' it appears in multiplication with.

     I'll split the derivation of the grads matrix using vectorized coding , into 2 parts, as final loss was also evaluated in steps.
1.)  Firstly, I'll work only with the " scores " matrix.
     Let's sum up the entire " scores " matrix. Now, each " w " in W matrix, occurs " m " no. of times in the scores sum , as each of the "m" x-vectors
     get multiplied with the same column of weights matrix in W, exactly once.

     So, now if we partially differentiate this sum w.r.t. each w[i][j] :- i'th feature's weight, of j'th class , we'll get the sum of i'th feature of each 'x' across
     all 'm' training sets. 

2.)  Now, we also have to rule out the elements in the final loss sum, that vanish due to max(  0, s(j)-s(i)+1  )
     So, for that, we'll take inner product of X.T with "the_first_ones_matrix".


          ''' 
  #2.)
  dw_matrix_part_one  =  (X.T).dot(  the_first_ones_matrix  )   # ----> (n+1,m)*(m,c) = (  n+1,c   )
  
  #3. )
  '''Logic behind the following steps. Finally, we also have to take care of the the subtractions of scores of actual class from the rest of the class
     scores for the same matrix. So, once again, we need to how many times each 'w' occurs with a negative sign, and with which 'x'. And then, we have to sum up
     those 'x'. Now, I'll create a separate matrix for that and subtract it from dw_matrix_part_one.
     Each 'w' of the particular class which is in 'y' will occur (c-1)*(no. of times the class to which that w belongs, occurs in 'y' vector).
     And, hence, we need to filter out certain 'x' vectors of the total 'm' vectors.
     Hence, the steps - 
     zeros_and_ones_matrix[ range(num_training),list(y) ] = 1
     dw_matrix_part_two =  ( num_classes-1)*(X.T).dot(  zeros_and_ones_matrix  )
     


      '''
  
  zeros_and_ones_matrix = np.zeros( (num_train, num_classes) ) 
  zeros_and_ones_matrix[ range(num_train),list(y) ] = 1
  
  
  dw_matrix_part_two =  (num_classes-1)*(X.T).dot(  zeros_and_ones_matrix  )


  # AND THE FINAL STEP
  
  dW_without_reg = dw_matrix_part_one  - dw_matrix_part_two

  dW_with_reg = dW/num_train + reg*W

  dW = dW_with_reg
  loss = loss_with_reg
  
  return loss, dW
