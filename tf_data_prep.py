import tensorflow as tf

def get_datasets(data_dir, img_size=(224, 224), batch_size=32):
    train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training",
        seed=42, image_size=img_size, batch_size=batch_size)
    
    val = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation",
        seed=42, image_size=img_size, batch_size=batch_size)

    norm = tf.keras.layers.Rescaling(1./255)
    return train.map(lambda x, y: (norm(x), y)), val.map(lambda x, y: (norm(x), y))
