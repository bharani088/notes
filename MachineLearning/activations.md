> Ref: <https://en.wikipedia.org/wiki/Activation_function>

non-linear transformation

# functions of one fold x

## Logistic function (sigmoid curve)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/faaa0c014ae28ac67db5c49b3f3e8b08415a3f2b)

(0,1)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/50a861269c68b1f1b973155fa40531d83c54c562)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Activation_logistic.svg/120px-Activation_logistic.svg.png)

## ReLU (Rectified linear unit)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/1d25c25758581789c97cdf80d52bf82bbfd0f237)

[0,+∞)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2b7ae55ba14c7ab5d170d0b484465b670bb38823)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/120px-Activation_rectified_linear.svg.png)

> recent work has shown sigmoid neurons to be less effective than rectified linear neurons. The reason is that the gradients computed by the backpropagation algorithm tend to diminish towards zero as activations propagate through layers of sigmoidal neurons, making it difficult to optimize neural networks using multiple layers of sigmoidal neurons.


## TanH (Hyperbolic tangent)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/48851c215e3c5b9dac76a25d4c358b9a2f7c137f)

(-1,1)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/b371c445bf1130914d25b1995d853ac0e27bc956)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Activation_tanh.svg/120px-Activation_tanh.svg.png)



# not functions of a single fold x

## Softmax

Softmax函数，或称归一化指数函数，是逻辑函数(Logistic function)的一种推广。它能将一个含任意实数的K维的向量z的“压缩”到另一个K维实向量σ(z)中，使得每一个元素的范围都在(0,1)之间，并且所有元素的和为1。

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/6d7500d980c313da83e4117da701bf7c8f1982f5)
for i = 1,...,J

(0,1)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/81a8feb8f01aaed053c103113e3b4917f936aef0)

>例子：
>输入向量[1,2,3,4,1,2,3]对应的Softmax函数的值为[0.024,0.064,0.175,0.475,0.024,0.064,0.175]。输出向量中拥有最大权重的项对应着输入向量中的最大值“4”。这也显示了这个函数通常的意义：对向量进行归一化，凸显其中最大的值并抑制远低于最大值的其他分量。