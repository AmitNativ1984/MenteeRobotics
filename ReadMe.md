Model:

You are going to implement a quantized variant of an auto-encoder architecture. The embedding tensor that is outputted by the encoder is quantized by finding its nearest neighbor in a dictionary of tensors. This quantized embedding tensor is then forwarded through a decoder.

Encoder e:

    Input: a batch of B input tensors, where each tensor is an image - [B, 3, H, W].

    Architecture: The feature extractor of ResNet 18

    Output: a batch of B tensors, where each tensor is D-dimensional - [B, D].


Quantizer :

    Initialize randomly a dictionary of tensors by a [K, D]  matrix, i.e. K tensors where each tensor is D-dimensional. These tensors are defined as learnable parameters.

    Given the embedding tensor that is outputted by the encoder e, find the nearest tensor in the dictionary matrix (in terms of the euclidean distance) and pass it as the input to the decoder.


Decoder d:

    Input: a batch of B input tensors, where each tensor is D-dimensional, i.e.  [B, D]

    Architecture: MLP

    Output: a batch of B tensors, where each tensor is an image - [B, 3, H, W].


Example:

Let x be an input image tensor - [B, 3, H, W].

x is forwarded through the encoder e, i.e. compute  e(x) - of size  [B, D]

Let q (of size [B,D]) be the nearest neighbor of e(x) in the tensors dictionary of the quantizer.

Forward q through the decoder, i.e. compute d(q) - of size [B, 3, H, W]

Dataset: CIFAR 10

Optimization:

The architecture should be trained end-to-end using the SGD optimizer.

You should implement the following two losses (both are applied side by side):

    Minimize the reconstruction loss (MSE loss) between the decoder’s output and the input to the encoder. We would like to train the auto-encoder end-to-end, but the quantization that is done in the quantizer has no real defined gradient for this loss. Therefore, we use an approximation and back-propagate the same gradients from the decoder’s input to the encoder’s output.


    The optimization of the tensors dictionary in the quantizer is done by minimizing the MSE loss between the output of the encoder, i.e. e(x) , and its nearest neighbor in the tensors dictionary. In contrast to the loss in (1), the gradients computed for loss (2) must not be back-propagated to the encoder. 