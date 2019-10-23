# BME590-Perceptual-Loss-Style-Transfer

This code implements style transfer from the paper **[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)** using Tensorflow 2.0 and Keras.

In this repository we provide three notebook files to perform the following:
- Transform an image using our pre-trained models in `Transform.ipynb`
- Train a new model using your style image of choice `Train.ipynb`
- Examine possible medical applications of style transfer in ultrasound images `Ultrasound.ipynb`

### Example outputs
#### Input image
<img src="images/input/chicago.jpg" width="256">

#### Styles
<img src="images/styles/starry_night_crop.jpg" width="256">
<img src="images/styles/udnie.jpg" width="256">

<img src="images/styles/feathers.jpg" width="256">
<img src="images/styles/the_scream.jpg" width="256">

#### Outputs
![chicago_starry](images/outputs/chicago_starry.jpg)
![chicago_udnie](images/outputs/chicago_udnie.jpg)
![chicago_feathers](images/outputs/chicago_feathers.jpg)
![chicago_scream](images/outputs/chicago_scream.jpg)
![duke](images/outputs/chapel_udnie.jpg)
