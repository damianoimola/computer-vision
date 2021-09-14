# computer-vision
This is a set of Computer Vision project

- [image-classification-animal]() is a model trained from alessiocorrado99's dataset about animals on Kaggle, named "animals10",

    If you run it in Google Colabe, be sure to make this steps, otherwise, skip this part
    ``` python
    from google.colab import files
    !pip install -q kaggle
    ```
    make a directory to store **kaggle.json** file
    ``` python
    ! mkdir ~/.kaggle
    ! cp kaggle.json ~/.kaggle/
    ! chmod 600 ~/.kaggle/kaggle.json
  
    # download the dataset inside colab folder
    !kaggle datasets download -d alessiocorrado99/animals10
  
    # unzip dataset
    !unzip /content/animals10.zip
    ```
- [image-recognition-mnist]() is a model built on digit mnist dataset. It is made with tensorflow eager mode.

    Just run "main" method, the dataset it needs is stored in Google Clous Storage Bucket, so easly retreivable, with the following command
    ``` python
    dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
    ``` 
    thanks to the attribute ```try_gcs=True```
    
