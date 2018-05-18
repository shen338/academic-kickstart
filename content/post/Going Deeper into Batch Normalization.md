+++
title = "Going Deeper in Batch Normalization" 
date = 2018-05-17T21:00:00
math = true
highlight = true

# List format.
#   0 = Simple
#   1 = Detailed
list_format = 1


tags = ["batch norm", "deep learning", "tensorflow"]
summary = "Go through the theory of batch normalization and its application"


# Optional featured image (relative to `static/img/` folder).
[header]
image = ""
caption = ""


+++

This article will thoroughly explain batch normalization in a simple way. 
I wrote this article after getting failed an interview because of detailed batchnorm related question. 
I will start with why we need it, how it works, then how to fuse it into conv layer, and finally how to implement it in tensorflow.   

Here is the original paper about batch normalization on Arxiv:   
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). 

### Interpretation and Advantage of Batch Norm   
Of course, batch norm is used to normalize the input for certain layer. We can think it in this way: if some of our input image have a scale between 0-1 while others are 
between 1-1000. It is better to normalize them before training. We can apply the same idea to the input of every layer input. 
There are several advantages to use batch norm:    
 
1. {{< hl >}}Batch norm can reduce the covariance shift{{< /hl >}}. For example, we train a model to classify cat and flowers.
And the training data of cat are all black cats. In this way, the model won't work because it can only 
classify the distribution of black cat and flowers. What batch norm does is to reduce this kind of error and make the 
input shift, like reduce the difference between black cat and other cats. And the same thing also applies to 
every layer in the neural network. Batch norm can reduce the shift around of previous output and make 
the training of next layers easier. 
2. {{< hl >}}Batch norm can remove the linear interactions in output{{< /hl >}}. In this way, linear layers would be useless, because they cano only have effect on 
linear component.  In a deep neural network with nonlinearactivation functions, the lower layers can perform nonlinear transformations of the data, so they remain useful. 
Batch normalization acts to standardize only the mean and variance of each unit in order to stabilize learning, but it allows therelationships 
between units and the nonlinear statistics of a single unit to change.
3. {{< hl >}}Batch normalization can greatly speed up training process{{< /hl >}}. Batch normalization accelerates training by requiring less iterations to 
converge to a given loss value. This can be done by using higher learning rates, but with smaller learning rates you can still see an improvement.   
Batch normalization also makes the optimization problem "easier", as minimizing the covariate shift avoid lots of plateaus where the loss stagnates
 or decreases slowly. It can still happen but it is much less frequent.  
4. {{< hl >}}Batch norm also has some regularization effect{{< /hl >}}. Every mini-batch is a biased sample from the total dataset.
When doing batch norm, we will subtract mean and divide it by variance. This can also be treated as add 
noise to data. Similar to regularization techniques like dropout, network can gain some regularization 
from this. But this effect is quite minor.   


### Algorithm and implementation
>Here is the algorithm diagram batch norm. 
>![Batch norm algorithm](/img/batch_norm_fp.png)    

Nothing fancy but extremely practical algorithm. One thing has to mention, {{< hl >}}the learnable variables
$\gamma$ and $\beta$. The deep learning book gives clear explaination about this{{< /hl >}}. Normalizing the mean and deviation of a unit can 
reduce the expressive power of a neural network. In this way, it is common to multiply the normalized result with $\gamma$ and add $\beta$. For exmaple, 
if we have sigmoid activation afterwards, the network may don't want the output lies in the near linear part of sigmoid. With $\gamma$ and $\beta$,
the network has the freedom to shift whatever it wants. 

>This is a new parametrization can represent the same family of functions of the input as the old parametrization, but the new parametrization
 has different learning dynamics. In the old parametrization, the mean of H was determined by a complicated interaction between the parameters 
 in the layers below H. In the new parametrization, the mean of $y=\gamma x + \beta$ is determined solely by $\gamma$. The new parametrization 
 is much easier to learn with gradient descent.    -- Deep Learning Book


{{< hl >}}At test time, we need the mean and variance directly. So, the method is using an exponentially weighted average across mini-batches. {{< /hl >}}
We have $ x_1, x_2, ... ,x_i $ outputs from different mini-batches. What we do is put expotential 
weight on previous processed mini-batches. The calculation is quite simple: 
$$Mean\_{running} = \mu * Mean\_{running} + (1.0 - \mu) * Mean\_{sample}$$
$$Var\_{running} = \mu * Var\_{running} + (1.0 - \mu) * Var\_{sample}$$
And we use running mean and var to calculate batchnorm.    
Alternatively, we can first calculate the total mean and variance of total test dataset. But
this exponential weighted method are more popular in practice.   

And last but not least, the code for forward and backward pass(from my cs231n homework): 
```python
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
       
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_stand = (x - sample_mean.T) / np.sqrt(sample_var.T + eps)

        out = x_stand * gamma + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * sample_mean
        running_var = momentum * running_var + (1.0 - momentum) * sample_var

        cache = (sample_mean, sample_var, x_stand, x, gamma, beta, eps)

       
    elif mode == 'test':
        

        x_stand = (x - running_mean) / np.sqrt(running_var)
        out = x_stand * gamma + beta

        
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

```
```python
def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    

    sample_mean, sample_var, x_stand, x, gamma, beta, eps = cache
    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_stand * dout, axis=0)
    dx = (1. / N) * gamma * (sample_var + eps)**(-1. / 2.) * (
         N * dout - np.sum(dout, axis=0) - (x - sample_mean) * (
         sample_var + eps)**(-1.0) * np.sum(dout * (x - sample_mean), axis=0))


    return dx, dgamma, dbeta

```

During training, the moving_mean and moving_variance need to be updated. 
By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op.
So, the template is: 
```python 
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```
And also tensorflow official evaluate function (classification model zoo): 
```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')
```

### Improvements and Alternatives
#### Batch norm fused with Convolution

This is the question makes me fail the interview. Actually, there is no magic stuff about fused batch normalization. Just mathematically 
calculate two layers together and treat them as one layer in forward and backward pass:    

Conv: 
$$convout(N_i,C\_{out})=bias(C\_{out})+\sum\_{k=1}^{C} weight(C\_{out},k)*input(N_i,k)$$
Batch Norm: 
$$bnout(N_i,C\_{out})=(convout(N_i,C\_{out})-\mu)/\sqrt{((\epsilon + \sigma)^2)}$$
$$bnout(N_i,C\_{out})=\gamma*bnout(N_i,C\_{out})+\beta$$
Here, $convout(N_i,C\_{out})$ means the $N_ith$ sample in the $C\_{out}$ channel. Same notation applies to input and bnout. 
$weight(C\_{out},k)$ is the conv kernel corresponding to $C\_{out}$. And $\epsilon, \sigma, \mu, \gamma, \beta$ are the same as above.    
After fusion, the total calculation becomes:    

$$out(N_i,C\_{out})=\gamma*(bias(C\_{out})+\sum\_{k=1}^{C} weight(C\_{out},k)*input(N_i,k)/\sqrt{((\epsilon + \sigma)^2)}+\beta$$
In this way, the weight and bias of fused conv layer is: 

$$bias = \gamma*(bias(C\_{out})$$
$$weight = weight(C\_{out},k)/\sqrt{((\epsilon + \sigma)^2)}+\beta$$

We can use the fused bias and weight in previous conv layers. We can drop the intermediate result between conv and batch norm using this method, 
which can save up to 50% memory and a minor increase of training time. 


#### Layer normalization

Just understand from its name, layer normalization. Instead of using a batch of data to produce $\mu and \sigma$ at every location, 
It uses all the neuron activations in one layer to produce $\mu and \sigma$. 
This method is especially useful when not using mini-batch like RNN, where batch norm cannot be used. But its performance in convs layers are not as good
as batch norm. 

#### Instance Normalization

#### Group Normalization

#### Other normalization techniques*


Recurrent Batch Normalization (BN) (Cooijmans, 2016; also proposed concurrently by Qianli Liao & Tomaso Poggio, but tested on Recurrent ConvNets, 
instead of RNN/LSTM): Same as batch normalization. Use different normalization statistics for each time step. You need to store a set of mean and 
standard deviation for each time step.

Batch Normalized Recurrent Neural Networks (Laurent, 2015): batch normalization is only applied between the input and hidden state, but not between 
hidden states. i.e., normalization is not applied over time.

Streaming Normalization (Liao et al. 2016) : it summarizes existing normalizations and overcomes most issues mentioned above. It works well with 
ConvNets, recurrent learning and online learning (i.e., small mini-batch or one sample at a time):

Weight Normalization (Salimans and Kingma 2016): whenever a weight is used, it is divided by its L2 norm first, such that the resulting weight has 
L2 norm 1. That is, output y=x*(w/|w|), where x and w denote the input and weight respectively. A scalar scaling factor g is then multiplied to the 
output y=y*g. But in my experience g seems not essential for performance (also downstream learnable layers can learn this anyway).

Cosine Normalization (Luo et al. 2017): weight normalization is very similar to cosine normalization, where the same L2 normalization is applied 
to both weight and input: y=(x/|x|)*(w/|w|). Again, manual or automatic differentiation can compute appropriate gradients of x and w.



Here are some great reference materials:  
1. [Deep Learning Book, Chapter 8.7.1](http://www.deeplearningbook.org/contents/optimization.html)   
2. [Stackoverflow: How could I use Batch Normalization in TensorFlow?](https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow)    
3. [Explaination on Covariance Shift](http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html)    
4. [Tensorflow batch normalization docs](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization)    
5. [Various Normalization Techniques in Deep Learning](https://datascience.stackexchange.com/questions/12956/paper-whats-the-difference-between-layer-normalization-recurrent-batch-normal)