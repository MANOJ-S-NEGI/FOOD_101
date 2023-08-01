# Dataset: FOOD_101 
dated: July 2023


### Project Description:

```
* 🔑 Programming Language: python
* 🔑 Web framework:  FastAPI
```

This dataset consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images.
 On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels.
All images were rescaled to have a maximum side length of 512 pixels.
the model achieved a validation  accuracy of 74% and an accuracy of 94% with the transfer learning approach EfficientNetB1.

![effiientb1](https://github.com/MANOJ-S-NEGI/FOOD_101/assets/99602627/8d903e57-0291-4466-9b59-971a267bd591)

## Model Predictions:
![effiientb1_output](https://github.com/MANOJ-S-NEGI/FOOD_101/assets/99602627/c464366f-59d7-4875-95eb-5ab4ea84f15a)

## Wrong Prediction by the model:
![effiientb1_wrong_output](https://github.com/MANOJ-S-NEGI/FOOD_101/assets/99602627/9517864e-d5ad-40cf-99c1-45209f45d098)


