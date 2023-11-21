# Paper - 2023-10-17

- [Brain tumor image generation using an aggregation of GAN models with style transfer](https://www.semanticscholar.org/paper/Brain-tumor-image-generation-using-an-aggregation-Mukherkjee-Saha/ce69e6b8b33414614285875d694532d6e4e96224)
- **Tags:** #GAN #Medical #data_augmentation 

## Main Information

| Paper Name                                                                          | Journal             | Citations | Date      | Model                | Metric | Dataset | Citations |
| ----------------------------------------------------------------------------------- | ------------------- | --------- | --------- | -------------------- | ------ | ------- | --------- |
| Brain tumor image generation using an aggregation of GAN models with style transfer | #scientific_reports | 20        | june 2022 | #GAN #style_transfer |        |         |           |


## Method


**Architecture:**
AGGrGAN, which consists in:

- To generate synthetic MRI scans of brain tumors:
	- Aggregation of 2 variants of Deep Convolutional GAN (DCGAN)
	- Wasserstein GAN (WGAN)
- Further is applied the style transfer technique to enhance image resemblance


**Work Done:**

- A novel aggregation method, called AGGrGAN, is proposed to combine synthesized images obtained from diferent GAN models. 
- An attempt is made to capture shared information from the latent representation of the generated images by the GAN models. 
- Style transfer is performed on the aggregated image to encapsulate the localized information of the source images. 
- Pixel wise aggregation of images has been performed, where the weights assigned to the images depend on what extent the corresponding pixel lies in the edge region. 
- The experiments have been done on two publicly available datasets—(i) brain tumor dataset34, (ii) BraTS 2020 dataset16, and the obtained results are satisfactory when measured in terms of some standard metrics.
## Experiments

### Datasets

**Scans:** Magnetic Resonance Imaging (MRI) - é o método de scaneamento de melhor performance

Two publicly available datasets:
- the brain tumor dataset 
- the Multimodal Brain Tumor Segmentation Challenge (BraTS) 2020 dataset.

## Metrics

Structural Similarity Index Measure (SSIM) scores

## Results

| Dataset                 | SSIM Score |
| ----------------------- | ---------- |
| the brain tumor dataset | 0.57       |
| BraTS 2020              | 0.83       | 

## Conclusions
