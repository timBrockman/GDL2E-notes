# Generative Deep Learning Notes

Notes based on [Generative Deep Learning 2nd Edition](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/) by [David Foster](https://github.com/davidADSP)

Here is the book's Repo: [https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition)

Here's Machine Learning Street Talk inteview with the Author [https://www.youtube.com/watch?v=CTA7-Bsa9U4](https://www.youtube.com/watch?v=CTA7-Bsa9U4)

 - [Part 1 Intro to Generative Deep Learning](#Part-1-Intro-to-Generative-Deep-Learning)
   -  [CH 1 Generative Modeling](#CH-1-Generative-Modeling)
   -  [CH 2 Deep Learning](#CH-2-Deep-Learning)
 - [Part 2 Methods](#Part-2-Methods)
   -  [CH 3 Variational Autoencoders](#CH-3-Variational-Autoencoders)
   -  [CH 4 Generative Adversarial Networks](#CH-4-Generative-Adversarial-Networks)
   -  [CH 5 Autoregressive Models](#CH-5-Autoregressive-Models)
   -  [CH 6 Normalizing Flow Models](#CH-6-Normalizing-Flow-Models)
   -  [CH 7 Energy-Based Models](#CH-7-Energy-Based-Models)
   -  [CH 8 Diffusion Models](#CH-8-Diffusion-Models)
 - [Part 3 Applications](#Part-3-Applications)
   -  [CH 9 Transformers](#CH-9-Transformers)
   -  [CH 10 Advanced GANs](#CH-10-Advanced-GANs)
   -  [CH 11 Music Generation](#CH-11-Music-Generation)
   -  [CH 12 World Models](#CH-12-World-Models)
   -  [CH 13 Multimodal Models](#CH-13-Multimodal-Models)
   -  [CH 14 Conclusion](#CH-14-Conclusion)
  
## Part 1 Intro to Generative Deep Learning
### CH 1 Generative Modeling
 - A. What is Generative Modeling
   - Generative vs. Discriminative Modeling
   - The Rise of Generative Modeling
   - Generative Modeling and AI 
 - B. Our first Generative Model
   - Hello World!
   - The Generative Modeling Framework
   - Representational Learning 
 - C. Core Probability Theory 
 - D. Generative Model Taxonomy
 - E. The Generative Deep Learning Code Base
   - Cloning the Repository
   - Using Docker
   - Running on a GPU 
 - Summary
### CH 2 Deep Learning
 - A. Data for Deep Learning
 - B. Deep Neural Networks
   - What is a Neural Network?
   - Learning High-Level Features
   - TensorFlow and Keras
 - C. Multilayer Perceptron (MLP)
   - Preparing the Data
   - Building the Model
   - Compiling the Model
   - Training the Model
   - Evaluating the Model
 - D. Convoluted Neural Networks
   - Convolutional Layers
   - Batch Normalization
   - Dropout
   - Building the CNN
   - Training and Evaluating the CNN
 - E. Summary

## Part 2 Methods
### CH 3 Variational Autoencoders
 - A. Autoencoders
   - The Fashion-MNIST Dataset
   - The Autoencoder Architecture
   - The Encoder
   - The Decoder
   - Joining the Encoder to the Decoder
   - Reconstructing Images
   - Visualizing the Latent Space
   - Generating New Images 
 - B. Variational Autoencoders
   - The Encoder
   - The Loss Function
   - Training the Variational Autoencoder
   - Analysis of the Variational Autoencoder 
 - C. Exploring Latent Space
   - The CelebA Dataset
   - Training the Variational Autoencoder
   - Analysis of the Variational Autoencoder
   - Generating New Faces
   - Latent Space Arithmetic
   - Morphing Between Faces
 - D. Summary
### CH 4 Generative Adversarial Networks
 - A. Deep Convolutional GAN (DCGAN)
   - The Bricks Dataset
   - The Discriminator
   - The Generator
   - Training the DCGAN
   - Analysis of the DCGAN
   - GAN Training: Tips and Tricks 
 - B. Wasserstein GAN with Gradient Penalty (WGAN-GP)
   - Wasserstein Loss
   - The Lipschitz Constraint
   - Enforcing the Lipschitz Contstraint
   - The Gradient Penalty Loss
   - Training the WGAN-GP
   - Analysis of the WGAN-GP
 - C. Conditional GAN (CGAN)
   - CGAN Architecture
   - Training the CGAN
   - Analysis of the CGAN 
 - D. Summary
### CH 5 Autoregressive Models
 - A. Long Short-Term Memory Network (LSTM)
   - The Recipes Dataset
   - Working with Text Data
   - Tokenization
   - Creating the Training Set
   - The LSTM Architecture
   - The Embedding Layer
   - The LSTM Layer
   - The LSTM Cell
   - Training the LSTM
   - Analysis of the LSTM 
 - B. Recurrent Neural Network (RNN) Extensions
   - Stacked Recurrent Networks
   - Gated Recurrent Units
   - Bidirectional Cells
 - C. PixelCNN
   - Masked Convolutional Layers
   - Risidual Blocks
   - Training the PixelCNN
   - Analysis of the PixelCNN
   - Mixture Distributions
 - D. Summary
### CH 6 Normalizing Flow Models
 - A. Normalizing Flows
   - Change of Variables
   - The Jacobian Determinant
   - The Change of Variables Equation
 - B. RealNVP
   - The Two Moons Dataset
   - Coupling Layers
   - Training the RealNVP Model
   - Analysis of the RealNVP Model
 - C. Other Normalizing Flow Models
   - GLOW
   - FFJORD
 - D. Summary
### CH 7 Energy-Based Models
 - A. Energy-Based Models
   - The MNIST Dataset
   - The Energy Function
   - Sampling Using Langevin Dynamics
   - Training with Contrastive Divergence
   - Analysis of the Energy-Based Model
   - Other Energy-Based Models
 - B. Summary
### CH 8 Diffusion Models
 - A. Denoising Diffusion Models (DMM)
   - The Flowers Dataset
   - The Forward Diffusion Process
   - The Reparameterization Trick
   - Diffusion Schedules
   - The Reverse Diffusion Process
   - The U-Net Denoising Model
   - Training the Diffusion Model
   - Sampling from the Denoising Diffusion Model
   - Analysis of the Diffusion Model
 - B. Summary

## Part 3 Applications
### CH 9 Transformers
 - A. GPT
 - B. Other Transformers
 - C. Summary
### CH 10 Advanced GANs
 - A. ProGAN
 - B. StyleGAN
 - C. StyleGAN2
 - D. Other Important GANs
 - E. Summary
### CH 11 Music Generation
 - A. Transformers for Music Generation
 - B. MuseGAN
 - C. Summary
### CH 12 World Models
 - A. Reinforcment Learning
 - B. World Model Overview
 - C. Collecting Random Rollout Data
 - D. Training the VAE
 - E. Collecting Data to Train the MDN-RNN
 - F. Training the MDN-RNN
 - G. Training the Controller
 - H. In-Dream Training
 - I. Summary
### CH 13 Multimodal Models
 - A. DALLE2
 - B. Imagen
 - C. Stable Diffusion
 - D. Flamingo
 - E. Summary
### CH 14 Conclusion
 - A. Timeline of Generative AI
 - B. The Current State of Generative AI
 - C. The Future of Generative AI
 - D. Final Thoughts
