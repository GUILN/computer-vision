 2023-10-16

- **Tags:** #survey #GAN

## Ideas 💡

**Chore Idea:** Use GANs - Generative Adversarial Networks as a research subject

Using survey: [Recent Advances of Generative Adversarial Networks in Computer Vision]([https://www.semanticscholar.org/paper/Recent-Advances-of-Generative-Adversarial-Networks-Cao-Jia/a4af1d879281da729a824d5825a507ac9ec54b50](https://www.semanticscholar.org/paper/Recent-Advances-of-Generative-Adversarial-Networks-Cao-Jia/a4af1d879281da729a824d5825a507ac9ec54b50))

## Basic Idea Of GANs

![[GAN_architecture.png]]

- Training will not stop until `Nash Equilibrium` is achieved
- At the end of the training $D$ (descriminator Ideally will not be able to distinguish real samples and generated samples)

### Key points of GAN

- Uses cross-entropy loss function
- 
### Challenges of GAN

- Gradient Disappearance
- Difficulty in training 
	- Discriminator not being well trained
	- It's hard to tell when the discriminator is well trained
	- generated distribution is difficult to overlap with real distribution
- Poor diversity

### GANs Architectures

- DCGAN - Deep Convolutional GAN
- CGAN - Conditional GAN
- WGAN - Wasserstein GAN
- WGAN-GP - WGAN with gradient penalty
- EBGAN - Energy Based GAN
- BEGAN - Boundary Equilibrium GAN
- InfoGAN - Information GAN
- LSGAN - Least Squares GAN
- ACGAN - Auxiliary Classifier GAN
- DRGAN - Degenerate Avoided GAN
- SNGAN - Spectral Normalization GAN
- JR-GAN - Jacobian Regularization GAN
- SAGAN - Self-Attention GAN
- STACKGAN - Gan for Text Mutual generation (generate img from text)

**DCGAN:**
- uses strided convolutions on the discriminator and fractional-strided convolutions on the generator to replace pooling layers

**LSGAN:** 
- Replaces the loss function (cross entropy) with least square loss

## Metrics 

- Qualitative comparisons: Compared by eye
	- Diversity  
	- Image clearness
- Quantitative comparisons: 
	- IS (inception score) - Evaluates quality of generated examples
	- FID (Fetched Inception Distance) 
		- Considered to be the best evaluation criteria

## Datasets

- MNIST
- Fashion-MNIST
## Applications

- Text-Mutual Generation (Generate images from texts)
- Style Transfer and Image Translation
- Generate High Quality Samples

## Related Papers | Current State of Art

### Related Papers

### Current State of Art

## Future Research Opportunities


