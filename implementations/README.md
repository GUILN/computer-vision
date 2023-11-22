 [Lung cancer CT image generation from a free-form sketch using style-based pix2pix for data augmentation](https://www.semanticscholar.org/paper/Lung-cancer-CT-image-generation-from-a-free-form-Toda-Teramoto/85ea8a4e2cffc1e18cb07487bd76de91c92ebbdb)

 ## 1. Introduction


 ## 2. Method

 1. Preprocessing
    1. Use cany edge detection to simulate the free-form sketch
    2. Use the cany edge detection result as the input of the generator and the original CT image as the ground truth
 2. Training
    1. Use the style-based pix2pix model
    2. Use the L1 loss and the adversarial loss
    3. Use the Adam optimizer

### 2.1 Style-based pix2pix
1. Generator
   1. U-NET
2. Discriminator
   1. PatchGAN

 ## Dataset

[LUNA16](https://luna16.grand-challenge.org/)
