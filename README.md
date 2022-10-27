# Phase retrieval from diffracted image

Source code for the post https://jordanhay.com/blog/2022/07/phase-retrieval

This is used to determine what an object looks like from the measurement of diffraction of a planar wave around the object. This shows up in experiments where X-rays are diffracted around crystals of interest and the diffracted rays measured with CCDs. The measured diffraction corresponds to the Fourier Modulus of the object.

This specific example takes a known image and computes its Fourier modulus (the diffraction of the image). It then produces a noisy 'guess' of what the image may look like and iteratively applies a difference map to the guess --- arriving at an approximation of the original image. 

![Example plot](https://github.com/JHay0112/phase-retrieval/blob/main/img/bad-guess-plot.png?raw=true)

Based on the following

[1] V. Elser, I. Rankenburg and P. Thibault, "Searching with Iterated Maps," 
  Proceedings of the National Academy of Sciences - PNAS, vol. 104, (2), pp. 418-423, 2007.

[2] V. Elser, "The Mermin Fixed Point," Foundations of Physics, vol. 33, (11), pp. 1691, 2003.

Thank you Joe for teaching me about this :D
