> ref: <http://neuralnetworksanddeeplearning.com/>

# Using neural nets to recognize handwritten digits

## Perceptrons

A perceptron takes several binary inputs, x1,x2,…x1,x2,…, and produces a single binary output:

![](http://neuralnetworksanddeeplearning.com/images/tikz0.png)

A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence.

w and x are vectors whose components are the weights and inputs, respectively, the bias is a measure of how easy it is to get the perceptron to output a 1
```
output={0, if w⋅x+b≤0 ; 1, if w⋅x+b>0 }
```

I've described perceptrons as a method for weighing evidence to make decisions. Another way perceptrons can be used is to compute the elementary logical functions we usually think of as underlying computation, functions such as AND, OR, and NAND.

## Sigmoid neurons

If it were true that a small change in a weight (or bias) causes only a small change in output, then we could use this fact to modify the weights and biases to get our network to behave more in the manner we want.

The problem is that this isn't what happens when our network contains perceptrons. In fact, a small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip, say from 0 to 1.

We can overcome this problem by introducing a new type of artificial neuron called a sigmoid neuron. Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output.

Just like a perceptron, the sigmoid neuron has inputs, x1,x2,…x1,x2,…. But instead of being just 0 or 1, these inputs can also take on any values between 0 and 1.

Also just like a perceptron, the sigmoid neuron has weights for each input, w1,w2,…w1,w2,…, and an overall bias, bb. But the output is not 00 or 11. Instead, it's `σ(w⋅x+b)`, where σ is called the `sigmoid function`:
```
σ(z) ≡ 1 / (1+e^−z)
# z = w⋅x+b
Δoutput ≈ ∑j ( ∂output/∂wj Δwj + ∂output/∂b Δb )
```

![sigmoid function](https://ml4a.github.io/images/figures/sigmoid.png)

If σ had in fact been a `step function`, then the sigmoid neuron would be a perceptron

![step function](https://qph.ec.quoracdn.net/main-qimg-d223b378c4b7b3edcb4d4f61607f6bca)

## The architecture of neural networks

![](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

Somewhat confusingly, and for historical reasons, such multiple layer networks are sometimes called multilayer perceptrons or MLPs, despite being made up of sigmoid neurons, not perceptrons. 

Such networks are called **feedforward neural networks (前馈神经网络)**. This means there are no loops in the network - information is always fed forward, never fed back. 

**recurrent neural networks (循环神经网络)** is a class of artificial neural network where connections between units form a directed cycle. This allows it to exhibit dynamic temporal behavior. Unlike feedforward neural networks, RNNs can use their internal memory to process arbitrary sequences of inputs.

## A simple network to classify handwritten digits

![](http://neuralnetworksanddeeplearning.com/images/tikz12.png)

You might wonder why we use 1010 output neurons. A seemingly natural way of doing that is to use just 44 output neurons since 2^4=16.

The ultimate justification is empirical: we can try out both network designs, and it turns out that, for this particular problem, the network with 1010 output neurons learns to recognize digits better than the network with 44 output neurons.

If we had 44 outputs, then the first output neuron would be trying to decide what the most significant bit of the digit was. And there's no easy way to relate that most significant bit to simple shapes like those shown above. It's hard to imagine that there's any good historical reason the component shapes of the digit will be closely related to (say) the most significant bit in the output.

## Learning with gradient descent

To quantify how well we're achieving this goal we define a **cost function**:
```
C(w,b) ≡ 1/2n ∑_x ‖y(x)−a‖^2.
```

![](http://neuralnetworksanddeeplearning.com/images/valley.png)

consider gradient descent when CC is a function of just two variables
```
ΔC ≈ ∂C/∂v1 Δv1 + ∂C/∂v2 Δv2
∇C ≡ (∂C/∂v1,∂C/∂v2)^T
ΔC ≈ ∇C⋅Δv

Δv = −η∇C, where η is a small, positive parameter (known as the learning rate). 
v → v′ = v − η∇C
```

## Implementing our network to classify digits



## Toward deep learning

Still, the heuristic suggests that if we can solve the sub-problems using neural networks, then perhaps we can build a neural network for face-detection, by combining the networks for the sub-problems.

![](http://neuralnetworksanddeeplearning.com/images/tikz14.png)

It's also plausible that the sub-networks can be decomposed. 

![](http://neuralnetworksanddeeplearning.com/images/tikz15.png)

Ultimately, we'll be working with sub-networks that answer questions so simple they can easily be answered at the level of single pixels.




# How the backpropagation algorithm works

At the heart of backpropagation is an expression for the partial derivative `∂C/∂w` of the cost function CC with respect to any weight w (or bias b) in the network. The expression tells us how quickly the cost changes when we change the weights and biases.

## Warm up: a fast matrix-based approach to computing the output from a neural network

![](http://neuralnetworksanddeeplearning.com/images/tikz16.png)

```
zl ≡ wla(l−1) + bl  # 带权输入
al = σ(zl)
```

## The two assumptions we need about the cost function

```
C = 1/2n ∑‖y(x)-aL(x)‖^2
```

The first assumption we need is that the cost function can be written as an average 
`C=1/n ∑_x Cx`
over cost functions Cx for individual training examples, x.
This is the case for the quadratic cost function, where the cost for a single training example is
`Cx=1/2 ‖y−aL‖^2`.

The second assumption we make about the cost is that it can be written as a function of the outputs from the neural network:
![](http://neuralnetworksanddeeplearning.com/images/tikz18.png)

## The Hadamard product, s⊙t

使用 s ⊙ t 来表示按元素的 乘积。所以 s ⊙ t 的元素就是 (s ⊙ t)j = sjtj。

## The four fundamental equations behind backpropagation？？

反向传播其实是对权重和偏置变化影响代价函数过程的理解。最终极的含义其实就是计算偏
导数 ∂C/∂wl 和 ∂C/∂bl 。但是为了计算这些值，我们首先引入一个中间量，δl ，这个我们称为
在l层第j个神经元上的误差。

增加很小的变化 ∆zjl 在神经元的带权输入上，使得神经元输出由 σ(zjl ) 变成 σ(zjl + ∆zjl )。这
个变化会向网络后面的层进行传播，最终导致整个代价产生 ∂C/∂zlj ∆zl 的改变。

假设 ∂C 有一个很大的值(或正或负)。那么这个调皮⻤可以通过选择与 ∂C/∂zlj 相反符号的 ∆zl 来降低代价。
所以这里有一种启发式的认识， ∂C/∂zlj 是神经元的误差的度量。

定义 l 层的第 jth 个神经元上的误差 δjl 为:
```
δlj ≡ ∂C/∂zjl
```

### BP1 输出层误差的方程，δL: 
```
δLj = ∂C/∂aLj σ′(zLj)

# proof:
δLj = ∂C/∂zjL
= ∂C/∂aLj * ∂aLj/∂zLj
= ∂C/∂aLj σ′(zLj)
```
右式第一个项 ∂C/∂aLj 表示代价随着 jth 输出激活值的变化而 变化的速度。假如 C 不太依赖一个特定的输出神经元 j，那么 δLj 就会很小，这也是我们想要的 效果。右式第二项 σ′(zLj) 刻画了在 zLj 处激活函数 σ 变化的速度。

以矩阵形式重写方程:
```
δL = ∂C/∂aL ⊙ σ′(zL)

# proof:
δL = ∇aC ⊙ σ′(zL)
= ∂C/∂aL ⊙ σ′(zL)
```
这里 ∇aC 被定义成一个向量，∂C/∂aL (元素为∂C/∂aLj)，是 C 关于输出 激活值的改变速度。
在二次代价函数时，我们有 `∇aC = (aL − y)`，所以 `δL = (aL − y) ⊙ σ′(zL)`

### BP2 使用下一层的误差 δl+1 来表示当前层的误差 δl:
```
δl = ((wl+1)^T δl+1) ⊙ σ′(zl)

# proof:
δl = ∂C/∂al ⊙ σ′(zl)
= ∂C/∂zl+1 * ∂zl+1/∂al ⊙ σ′(zl)
= ((wl+1)^T δl+1) ⊙ σ′(zl)
```
其中 (wl+1)^T 是 l+1 层权重矩阵 wl+1 的转置。

### BP3 代价函数关于网络中任意偏置的改变率: 
```
∂C/∂blj = δlj

# proof:
∂C/∂blj = ∂C/∂zlj * ∂zlj/∂blj
= ∂C/∂zlj * 1
= δlj
```
这其实是，误差 δlj 和偏导数值 ∂C/∂blj 完全一致。

### BP4 代价函数关于任何一个权重的改变率: 
```
∂C/∂wljk = al−1k δlj = ain δout


# proof:
∂C/∂wljk = ∂C/∂zlj * ∂zlj/∂wljk
= al−1k δlj
```

![summary](http://neuralnetworksanddeeplearning.com/images/tikz21.png)

## Proof of the four fundamental equations (optional)

> see above proofs

## The backpropagation algorithm

The backpropagation equations provide us with a way of computing the gradient of the cost function.

* **Input** x: Set the corresponding activation a1 for the input layer.
* **Feedforward**: For each l=2,3,…,L compute `zl = wl*al−1 + bl` and `al = σ(zl)`.
* **Output error** δL: Compute the vector `δL = ∇aC ⊙ σ′(zL)`.
* **Backpropagate the error**: For each l=L−1,L−2,…,2 compute `δl = ((wl+1)Tδl+1) ⊙ σ′(zl)`.
* Output: The gradient of the cost function is given by `∂C/∂wljk = al−1k δlj` and `∂C/∂blj = δlj`.

这种反向移 动其实是代价函数是网络输出的函数的结果。为了理解代价随前面层的权重和偏置变化的规律， 我们需要重复作用链式法则，反向地获得需要的表达式。

## The code for backpropagation

```python
class Network(object):

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
```

## In what sense is backpropagation a fast algorithm?
仅仅把代价看做权重的函数 C = C(w)，期望计算某些权重 wj 的偏导数 ∂C/∂wj，一种近似的方法就是下面这种:
```
∂C/∂wj ≈ [C(w + εej) − C(w)] / ε
```
其中 ε > 0 是一个很小的正数，而 ej 是在第 j 个方向上的单位向量。同样方法也可以用 来计算 ∂C/∂b。

然后，遗憾的是，当你实现了之后，运行起来这样的方法非常缓慢。
对每个不同的权重wj 我们需要计算C(w+εej)来计算∂C/∂wj。

反向传播聪明的地方就是它确保我们可以同时计算所有的偏导数 ∂C/∂wj，仅仅使用一次前 向传播，加上一次后向传播。粗略地说，后向传播的计算代价和前向的一样 。所以反向传播总 的计算代价大概是两倍的前向传播。比起直接计算导数，显然反向传播有着更大的优势。

## Backpropagation: the big picture
wljk 到 C 的 路径，每个激活值的变化会导致下一层的激活值的变化，最终是输出层的代价的变化。
整个网络中存在很多 wl 可以传播而影响代价函数的路径。为了计算 C 的全部改变，我们就需要对 所有可能的路径进行求和。

这仅仅是一个神经元的激活值相对于其他神经元的激活值的偏
导数。从第一个权重到第一个神经元的变化率因子是 ∂al /∂wljk 。路径的变化率因子其实就是这
条路径上的众多因子的乘积。而整个的变化率 ∂C/∂wl 就是对于所有可能的从初始权重到最终
输出的代价函数的路径的变化率因子的和。

我们到现在所给出的东西其实是一种启发式的观点，一种思考权重变化对网络行为影响的方 式。让我们给出关于这个观点应用的一些流程建议。首先，你可以推导出公式中所有单独 的的偏导数显式表达式。只是一些微积分的运算。完成这些后，你可以弄明白如何用矩阵运算 写出对所有可能的情况的求和。这项工作会比较乏味，需要一些耐心，但不用太多的洞察。完 成这些后，就可以尽可能地简化了，最后你发现，自己其实就是在做反向传播!所以你可以将 反向传播想象成一种计算所有可能的路径变化率的求和的方式。或者，换句话说，反向传播就 是一种巧妙地追踪权重(和偏置)微小变化的传播，抵达输出层影响代价函数的技术。



# Improving the way neural networks learn

## The cross-entropy cost function (交叉熵代价函数)

我们通常是在犯错比较明显的时候学习的速度最快。但是我们已经看到了人工神经元在其犯错较大的情况下其实学习 很有难度。而且，这种现象不仅仅是在这个小例子中出现，也会在更加一般的神经网络中出现。

![一个单输入变量的的神经元](http://neuralnetworksanddeeplearning.com/images/tikz28.png)

我们会训练这个神经元来做一件非常简单的事:让输入 1 转化为 0。

```
C = (y-a)^2 / 2

# 其中我已经将 x = 1 和 y = 0 代入了
∂C/∂w = (a−y)σ′(z)x = aσ′(z)
∂C/∂b = (a−y)σ′(z) = aσ′(z)
```

代价函数的偏导数(∂C/∂w 和 ∂C/∂b)决定的速度学习。所以，我们在说“学习缓慢”时，实际上就是说 这些偏导数很小。

从sigmoid function图像看出当神经元输出接近 1 时，曲线变得非常平坦，
因此 σ′(z)就会变得非常小，∂C/∂w 和 ∂C/∂b会变得很小。这就是学习速度变慢的根源。

![一个包含若干输入变量的的神经元](http://neuralnetworksanddeeplearning.com/images/tikz29.png)

定义这个神经元的交叉熵代价函数，
```
C = −1/n ∑_x[ylna + (1−y)ln(1−a)]
```
其中 n 是训练数据的总数，求和是在所有的训练输入 x 上进行的，y 是对应的目标输出。

第一，它是非负的，C > 0。

第二，如果对于所有的训练输入 x，神经元实际的输出接近目标值，那么交叉熵将接近 0。

交叉熵是非负的，在神经元达到很好的正确率的时候会接近 0。

交叉熵代价函数有一个比二次代价函数更好的特性就是它避免了学习速度下降的问题。

当我们使用交叉熵的时 候，σ′(z) 被约掉了，所以我们不再需要关心它是不是变得很小。这种约除就是交叉熵带来的特 效。
```
∂C/∂wj = 1/n ∑_x xj(σ(z)−y)
∂C/∂b = 1/n ∑_x (σ(z)−y)
```
这是一个优美的公式。它告诉我们权重学习的速度受到 σ(z) − y，也就是输出中的误差的控 制。更大的误差，更快的学习速度。


### 柔性最大值(softmax)

柔性最大值的想法其实就是为神经网络定义一种新式的输出层。开始时和 S 型层一样的，首
先计算带权输入 `zLj = ∑_k wLjk aL−1k + bLj`。
不过，这里我们不会使用 S 型函数来获得输出。而是会在这一层上应用一种叫做柔性最大值函数在 zL 上。
第 j 个神经元的激活 值aLj 就是
```
aLj = e^zLj / ∑_k e^zLk
```
其中，分母中的求和是在所有的输出神经元上进行的。

当你增加 z4L 的时候，你可以看到对应激活值 aL4 的增加，而其他的激活值就在下降。
类似地，如果你降低 z4L 那么 aL4 就随之下降，而其它激活值则增加。
原因很简单，根据定义，输出的激活值加起来正好为 1 。
```
∑_j aLj = ∑_j e^zLj / ∑_k e^zLk = 1
```

柔性最大值层的输出是一些相加为 1 正数的集合。
换言之，柔性最大值层的输出可以被看做是一个概率分布。

在很多问题中，能够将输出激活值 aLj 理解为网络对于正确输出为 j 的概率的估计是非常方便的。


## Overfitting and regularization

检测过度拟合的明显方法是使用上面的方法 —— 跟踪测试数据集合上的准确率随训练变化 情况。如果我们看到测试数据上的准确率不再提升，那么我们就停止训练。当然，严格地说，这其实并非是过度拟合的一个必要现象，因为测试集和训练集上的准确率可能会同时停止提升。 当然，采用这样的策略是可以阻止过度拟合的。

我们会使用 validation_data 而不是 test_data 来防止过度拟 合。我们会为 test_data 使用和上面提到的相同的策略。我们在每个迭代期的最后都计算在 validation_data 上的分类准确率。一旦分类准确率已经饱和，就停止训练。这个策略被称为**提前停止**。
这个一般的策略就是使用 validation_data 来衡量不同的超参数(如迭代期， 学习速率，最好的网络架构等等)的选择的效果。我们使用这样方法来找到超参数的合适值。
当设置超参数时，我们想要尝试许多不同的超参数选择。如果我们 设置超参数是基于 test_data 的话，可能最终我们就会得到过度拟合于test_data 的超参数。
我们借助 validation_data 来克服这个问题。然后一旦获得了想要的超参数，最终 我们就使用 test_data 进行准确率测量。

这种寻找好的超参数的方法有时候被称为 **hold out** 方法，因为 validation_data 是从traning_data 训练集中留出或者“拿出”的一部分。

最为常用的规范化手段 —— 有时候被称为**权重衰减(weight decay)**或者 **L2 规范化**。L2 规范化的想法是增加一个额外的项到代价函数上， 这个项叫做**规范化项**。

规范化可以当做一种寻找小的权重和最小化原始的代价函数之间的折中。这两部分之间相对的重要性由 λ 的值来控制。

小的权重在某种程度上，意味着更低的复杂性，也就对数据给出 了一种更简单却更强大解释，因此应该优先选择。

更小的权重意味着网络的行为不会因为我们随便改变了一个输入而改变太大。
这会让规范化网络学习局部噪声的影响更加困难。将它看做是一种让单个的证据不会影响网络输出太多的方式。

## Weight initialization

之前的方式就是根据独立高斯随机变量来选择权重和 偏置，其被归一化为均值为 0，标准差 1。
这个方法工作的还不错，但是非常特别，所以值得去重新探讨它，看看是否能寻找一些更好的方式来设置初始的权重和偏置，这也许能帮助我们的 网络学习得更快。

假设我们使用训练输入 x，其中一半的输入神经元值为 1，另一半为 0。
z = ∑ wj xj + b。
z 其实有一个非常宽的高斯分布，完全不是非常尖的形状。 
|z| 会变得非常的大，即 z ≫ 1 或者 z ≪ −1。如果是这样， 隐藏神经元的输出 σ(z) 就会接近 1 或者 0。也就表示我们的隐藏神经元会饱和。
所以当出现这样的情况时，在权重中进行微小的调整仅仅会给隐藏神经元的激活值带来极其微弱的改变。而 这种微弱的改变也会影响网络中剩下的神经元，然后会带来相应的代价函数的改变。
结果就是， 这些权重在我们进行梯度下降算法时会学习得非常缓慢。

当然，类似的论证也适用于后面的隐藏层:如果后 面隐藏层的权重也是用归一化的高斯分布进行初始化，那么激活值将会非常接近 0 或者 1，学习 速度也会相当缓慢。

假设我们有一个有 n 个输入权重的神经元。我们会使用均值为 0 标准差为 1/√n 的高斯 随机分布初始化这些权重。也就是说，我们会向下挤压高斯分布，让我们的神经元更不可能饱 和。我们会继续使用均值为 0 标准差为 1 的高斯分布来对偏置进行初始化，后面会告诉你原因。 有了这些设定，带权和 z = ∑ wj xj + b 仍然是一个均值为 0，不过有尖锐峰值的高斯分布。
这样的一个神经元更不可能饱和，因此也不大可能遇到学习速度下降的问题。

## Handwriting recognition revisited: the code

初始化权重：使用均值为 0 而标准差为 1/√n，n 为对应的输入连接个数。

初始化偏置：使用均值 为 0 而标准差为 1 的高斯分布。

```python
def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                for x, y in zip(self.sizes[:-1], self.sizes[1:])]
```

## How to choose a neural network's hyper-parameters?

## Other techniques

# A visual proof that neural nets can compute any function
## Two caveats
## Universality with one input and one output
## Many input variables
## Extension beyond sigmoid neurons
## Fixing up the step functions
## Conclusion

# Why are deep neural networks hard to train?

在深度神经网络中使用基于梯度下降的学习 方法本身存在着内在不稳定性。这种不稳定性使得先前或者后面的层的学习过程阻滞。

至少在某些深度神经网络中，在我们在隐藏层 BP 的时候梯度倾向于变小。这意味着在前面的隐藏层中的神经元学习速度要慢于后面的隐藏层。 这儿我们只在一个网络中发现了这个现象，其实在多数的神经网络中存在着更加根本的导致这 个现象出现的原因。这个现象也被称作是消失的梯度问题 (vanishing gradient problem)

## The vanishing gradient problem
## What's causing the vanishing gradient problem? Unstable gradients in deep neural nets

let's consider the simplest deep neural network: one with just a single neuron in each layer. Here's a network with three hidden layers:

![](http://neuralnetworksanddeeplearning.com/images/tikz37.png)

```
∂C/∂b1 = σ′(z1) w2σ′(z2) w3σ′(z3) w4σ′(z4) ∂C/∂a4
```

```
# 我们看看 ∆b1 如何影响第一个神经元的输出 a1 的。
a1 = σ(z1) = σ(w1a0 + b1)
∆a1 ≈ ∂σ(w1a0 + b1)/∂b1 ∆b1 = σ′(z1)∆b1

# σ′(z1) 这项看起很熟悉:其实是上面关于 ∂C/∂b1 的表达式的第一项。
# 直觉上看，这项将偏置的改变 ∆b1 转化成了输出的变化 ∆a1。
# ∆a1 随之又影响了带权输入z2
z2 = w2 ∗ a1 + b2
∆z2 ≈ ∂z2/∂a1 ∆a1 = w2∆a1

# 将 ∆z2 和 ∆a1 的表达式组合起来，可以看到偏置 b1 中的改变如何通过网络传输影响到 z2。
∆z2 ≈ σ′(z1)w2∆b1
# 现在，又能看到类似的结果了:我们得到了在表达式 ∂C/∂b1 的前面两项。
```

除了最后一项，该表达式是一系列形如 wjσ′(zj) 的乘积。为了理解每个项的行为，先看看下 面的 sigmoid 函数导数的图像:
该导数在 σ′(0) = 1/4 时达到最高。现在，如果我们使用标准方法来初始化网络中的权重，那 么会使用一个均值为 0 标准差为 1 的高斯分布。因此所有的权重通常会满足 |wj | < 1。有了这些 信息，我们发现会有 wj σ′(zj ) < 1/4。并且在我们进行了所有这些项的乘积时，最终结果肯定会 指数级下降:项越多，乘积的下降的越快。 这里我们敏锐地嗅到了消失的梯度问题的合理解释。

我们想要知道权重wj 在训练中是否会增⻓。如果会，项wjσ′(zj)会不 会不在满足之前 wj σ′(zj ) < 1/4 的约束。事实上，如果项变得很大——超过 1，那么我们将不再 遇到消失的梯度问题。实际上，这时候梯度会在我们 BP 的时候发生指数级地增⻓。也就是说， 我们遇到了 **梯度爆炸** 的问题。

根本的问题其实并非是消失的梯度问题或者爆炸的梯度问题，而是在前 面的层上的梯度是来自后面的层上项的乘积。当存在过多的层次时，就出现了内在本质上的不 稳定场景。唯一让所有层都接近相同的学习速度的方式是所有这些项的乘积都能得到一种平衡。 如果没有某种机制或者更加本质的保证来达成平衡，那网络就很容易不稳定了。
简而言之，真实的问题就是神经网络受限于 **不稳定梯度的问题**。

## Unstable gradients in more complex networks

```
δl = Σ′(zl)(wl+1)TΣ′(zl+1)(wl+2)T ... Σ′(zL)∇aC
```

这里 Σ′(zl) 是一个对⻆矩阵，每个元素是对第 l 层的带权输入 σ′(z)。而 wl 是对不同层的权 值矩阵。∇aC 是对每个输出激活的偏导数向量。

这是更加复杂的表达式。不过，你仔细看，本质上的形式还是很相似的。主要是包含了更多 的形如 (wj)T Σ′(zj) 的对 (pair)。而且，矩阵 Σ′(zj) 在对⻆线上的值挺小，不会超过 1/4。由于 权值矩阵 wj 不是太大，每个额外的项 (wj)T σ′(zl) 会让梯度向量更小，导致梯度消失。更加一 般地看，在乘积中大量的项会导致不稳定的梯度，和前面的例子一样。

## Other obstacles to deep learning

本章，我们已经集中 于深度神经网络中基于梯度的学习方法的不稳定性。结果表明了激活函数的选择，权重的初始 化，甚至是学习算法的实现方式也扮演了重要的⻆色。当然，网络结构和其他超参数本身也是 很重要的。

# Deep learning

## Introducing convolutional networks

使用全连接层的网络来分类图像是很奇怪 的。原因是这样的一个网络架构不考虑图像的空间结构。例如，它在完全相同的基础上去对待相 距很远和彼此接近的输入像素。这样的空间结构的概念必须从训练数据中推断。

卷积神经网络采用了三种基本概念:**局部感受野(local receptive fields)**，**共享权重(shared weights)**，和**混合(pooling)**。

### 局部感受野(local receptive fields)

把输入看作是一个28 × 28方形排列的神经元。
和通常一样，我们把输入像素连接到一个隐藏神经元层。但是我们不会把每个输入像素连接
到每个隐藏神经元。
相反，我们只是把输入图像进行小的，局部区域的连接。第一个隐藏层中的每个神经元会连接到一个输入神经元的一个小区域，例如， 一个 5 × 5 的区域，对应于 25 个输入像素。
这个输入图像的区域被称为隐藏神经元的局部感受野。

每个 连接学习一个权重。而隐藏神经元同时也学习一个总的偏置。你可以把这个特定的隐藏神经元 看作是在学习分析它的局部感受野。
我们然后在整个输入图像上交叉移动局部感受野。对于每个局部感受野，在第一个隐藏层中
有一个不同的隐藏神经元。

![](http://neuralnetworksanddeeplearning.com/images/tikz44.png)
![](http://neuralnetworksanddeeplearning.com/images/tikz45.png)

如此重复，构建起第一个隐藏层。注意如果我们有一个 28 × 28 的输入图像，5 × 5 的局部感 受野，那么隐藏层中就会有 24 × 24 个神经元。

我显示的局部感受野每次移动一个像素。实际上，有时候会使用不同的**跨距**。

### 共享权重(shared weights)和偏置

我们打算对 24 × 24 隐藏神经元中的每一个使用相同的权重和偏置。
第 j, k 个隐藏神经元，输出为:
```
σ(b + ∑_l=0~4∑_m=0~4 w_l,m a_j+l,k+m)
```
这里 σ 是神经元的激活函数，可以是我们在前面章里使用过的 S 型函数。b 是偏置的共享 值。w_l,m 是一个共享权重的 5 × 5 数组。最后，我们使用 a_x,y 来表示位置为 x, y 的输入激活值。

人们有时把这个方程写成 `a1 = σ(b + w ∗ a0)`，其中 a1 表示出 自一个特征映射的输出激活值集合，a0 是输入激活值的集合，而 ∗ 被称为一个卷积操作

这意味着第一个隐藏层的所有神经元检测完全相同的特征，只是在输入图像的不同位置。
卷积网络能很好地适应图像的平 移不变性:例如稍稍移动一幅猫的图像，它仍然是一幅猫的图像。

因为这个原因，我们有时候把从输入层到隐藏层的映射称为一个**特征映射**。我们把定义特征 映射的权重称为**共享权重**。我们把以这种方式定义特征映射的偏置称为**共享偏置**。共享权重和 偏置经常被称为一个**卷积核**或者**滤波器**。

目前我描述的网络结构只能检测一种局部特征的类型。为了完成图像识别我们需要超过一个
的特征映射。所以一个完整的卷积层由几个不同的特征映射组成

![](http://neuralnetworksanddeeplearning.com/images/tikz46.png)

在这个例子中，有 3 个特征映射。每个特征映射定义为一个 5 × 5 共享权重和单个共享偏置 的集合。其结果是网络能够检测 3 种不同的特征，每个特征都在整个图像中可检测。

Let's take a quick peek at some of the features which are learned

![](http://neuralnetworksanddeeplearning.com/images/net_full_layer_0.png)

这 20 幅图像对应于 20 个不同的特征映射(或滤波器、核)。每个映射有一幅 5 × 5 块的图像 表示，对应于局部感受野中的 5 × 5 权重。白色权重小，暗色权重大。

> [卷积神经网络工作原理直观的解释？](https://www.zhihu.com/question/39022858)
滤波器跟卷积神经网络有什么关系呢。不如我们预想一个识别问题：我们要识别图像中的某种特定曲线，也就是说，这个滤波器要对这种曲线有很高的输出，对其他形状则输出很低，这也就像是神经元的激活。

相比全连接，共享权重和偏置的一个很大的优点是，它大大减少了参与的卷积网络的参数。

### 混合(pooling)

混合层(pooling layers)通常紧接着在卷积层之后使用。它要做的是简化从卷积层输出的信息。一个混合层取得从卷积层输出的每一个特征映射并且从它们准备一个凝缩的特征映射。
一个常⻅的混合的程序被称为**最大值混合(max-pooling)**。

我们可以把最大值混合看作一种网络询问是否有一个给定的特征在一个图像区域中的哪个地 方被发现的方式。然后它扔掉确切的位置信息。直观上，一旦一个特征被发现，它的确切位置 并不如它相对于其它特征的大概位置重要。

另一个常用的方法是 **L2 混合(L2 pooling)**。这里 我们取 2 × 2 区域中激活值的平方和的平方根，而不是最大激活值。

综合在一起

![](http://neuralnetworksanddeeplearning.com/images/tikz49.png)

## Convolutional neural networks in practice
## The code for our convolutional networks
## Recent progress in image recognition
## Other approaches to deep neural nets

**递归神经网络(RNN)**
> <https://zh.wikipedia.org/wiki/递归神经网络>

递归神经网络（RNN）是两种人工神经网络的总称。一种是时间递归神经网络（recurrent neural network），另一种是结构递归神经网络（recursive neural network）。

RNN一般指代时间递归神经网络。单纯递归神经网络因为无法处理随着递归，权重指数级爆炸或消失的问题（Vanishing gradient problem），难以捕捉长期时间关联；而结合不同的LSTM可以很好解决这个问题。

时间递归神经网络可以描述动态时间行为，因为和前馈神经网络（feedforward neural network）接受较特定结构的输入不同，RNN将状态在自身网络中循环传递，因此可以接受更广泛的时间序列结构输入。

## On the future of neural networks

