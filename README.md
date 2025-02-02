# IA048

This repository contains the final project for the course IA 048 (Machine Learning - Unicamp), developed during the COVID-19 pandemic. The goal of this project was to create a real-time facial mask detection solution using machine learning techniques and a webcam. The application was able to capture live images, track faces, and identify whether the person was wearing a mask or not.

The project is divided into two main parts:
1. **Model Training and Validation**: The first part contains the training and validation of the model, which was done using a notebook/Colab environment. During training, the model's weights are saved using the `ModelCheckpoint` callback, which stores the best weights based on the validation loss. These weights are saved in a file called `cp.ckpt`.

   ```python
   checkpoint_path = "model/cp.ckpt"
   checkpoint_dir = os.path.dirname(checkpoint_path)

   cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    monitor="val_loss",
                                                    save_weights_only=True,
                                                    verbose=0,
                                                    save_best_only=True,
                                                    period=1)
   ```
2. **Model Deployment**: The second part involves manually importing the trained model's weights (checkpoints) in order to run the model locally. To use the model with a webcam, the user should download the saved `cp.ckpt` file and load the weights using the following command: ```python model.load_weights('path/to/model/cp.ckpt')```. Once the weights are loaded, the model can be used in real-time to detect mask usage through the webcam.
