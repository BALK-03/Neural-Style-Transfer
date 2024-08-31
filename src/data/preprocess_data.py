import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt




def load_image(img_path):
    max_size = 512
    image = PIL.Image.open(img_path)
    
    image_np = np.array(image)
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_shape = tf.shape(image_tensor).numpy()
    
    largest_dim = max(image_shape[:-1])
    
    if largest_dim > max_size:
        scale = max_size / largest_dim
        new_shape = tf.cast(image_shape[:-1] * scale, tf.int32)
        image_tensor = tf.image.resize(image_tensor, new_shape)
    
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    # Add an extra dimension (1, height, width, channels)
    output = tf.cast(image_tensor, dtype=tf.uint8).numpy()
    
    return output



def show(image, title):
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)



def img_preprocess(img_path):
    image = load_image(img_path)
    # Applying specific preprocessing required by VGG19
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image




def deprocess_img(processed_img):
    copy_img = processed_img.copy()

    # (1, height, width, channels) -> (height, width, channels)
    if len(copy_img.shape) == 4:
        copy_img = np.squeeze(copy_img, 0)
    assert len(copy_img.shape) == 3

    # VGG19 preprocessing includes substructing the mean values of each channel,
    # those values are related to ImageNet dataset
    copy_img[:, :, 0] += 103.939
    copy_img[:, :, 1] += 116.779
    copy_img[:, :, 2] += 123.68
    # BGR -> RGB
    copy_img = copy_img[:, :, [2, 1, 0]]

    # Clips pixel values back to [0, 255]
    copy_img = np.clip(copy_img, 0, 255).astype('uint8')
    return copy_img