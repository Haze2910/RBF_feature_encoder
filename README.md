# RBF Feature Encoder
A Radial Basis Function kernel is defined as

$$
K(x, x') = \exp{\left(-\frac{\|x-x'\|^2}{2\sigma^2}\right)}
$$

where $x, x'\in\mathbb{R}^k$.
The RBF Encoder takes a number of centers $c_1,...,c_n$, with $c_i\in\mathbb{R}^k$ and encodes the input into

$$
y = \left(K(x, c_1), K(x, c_2), \cdots, K(x, c_n)\right)
$$

This repo is a simple implementation of the RBF encoding of a normalized batch of data points.
