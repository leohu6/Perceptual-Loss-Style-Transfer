# BME590-Perceptual-Loss-Style-Transfer
By Leo Hu and Chris Zhou

This code implements style transfer from the paper **[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)** using Tensorflow 2.0 and Keras.

In this repository we provide three notebook files to perform the following:
- Transform an image using our pre-trained models in `Transform.ipynb`
- Train a new model using your style image of choice `Train.ipynb`
- Examine possible medical applications of style transfer in ultrasound images `Ultrasound.ipynb`

### Example outputs
#### Input image
<img src="images/input/chicago.jpg" width="512">

#### Styles
<img src="images/styles/starry_night_crop.jpg" height="256">
<img src="images/styles/udnie.jpg" height="256">
<img src="images/styles/feathers.jpg" height="256">
<img src="images/styles/the_scream.jpg" height="256">

#### Outputs
<img src="images/outputs/chicago_starry.jpg" width="1024">
<img src="images/outputs/chicago_udnie.jpg" width="1024">
<img src="images/outputs/chicago_feathers.jpg" width="1024">
<img src="images/outputs/chicago_scream.jpg" width="1024">
<img src="images/outputs/chapel_udnie.jpg" width="1024">
