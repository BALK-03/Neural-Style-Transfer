
import os
import sys
import tensorflow as tf
import IPython.display as display

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from test_environment import check_environment
from src.data.preprocess_data import *


# Importing only the convolution layers of VGG19
model = tf.keras.applications.vgg19.VGG19(
    include_top = False,
    weights="imagenet"
)

# freezing the layers
model.trainable = False

# this layer cappture the overall structure of the image
content_layers = ['block5_conv2']

# these layers capture the style of the image
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

# getting the output of content and style layers
content_outputs = [model.get_layer(name).output for name in content_layers]
style_outputs = [model.get_layer(name).output for name in style_layers]

# building our model
transfer_model = tf.keras.models.Model(
    inputs = model.inputs,
    outputs = style_outputs + content_outputs
)

# freeze the layers 
transfer_model.trainable = False


def content_loss(content_features, generated_features):    
    total_loss = 0.0
    for content, generated in zip(content_features, generated_features):
        layer_loss = tf.reduce_mean(tf.square(content - generated))
        total_loss += layer_loss
    
    average_loss = total_loss / 2.0
    
    return average_loss


def gram_matrix(input_tensor):
    input_shape = tf.shape(input_tensor)
    
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    num_elements = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    gram_normalized = gram / num_elements
    return gram_normalized


def style_loss(style_features, generated_features):
    loss_values = []
    for style, generated in zip(style_features, generated_features):
        style_gram = gram_matrix(style)
        generated_gram = gram_matrix(generated)
        
        layer_loss = tf.reduce_mean(tf.square(style_gram - generated_gram))
        loss_values.append(layer_loss)
        
    total_loss = tf.add_n(loss_values) / tf.cast(len(style_features), tf.float32)
    return total_loss


def get_features(model, tensor):
    output = model(tensor)
    return {
        "style": output[:len(style_layers)],
        "content": output[len(style_layers):]
    }


def total_loss(style_loss , content_loss , style_weight = 1e-2 , content_weight = 1e4):
    return style_loss * style_weight + content_loss * content_weight


def clip(image, min_val, max_val):
    return tf.clip_by_value(image, clip_value_min=min_val, clip_value_max=max_val)


def gradient_descent(model, optimizer, gen_image, style_features,
                     content_features, style_weight, content_weight):
    with tf.GradientTape() as tape:
        tape.watch(gen_image)
        
        gen_image_style_features, gen_image_content_features = get_features(model, gen_image).values()
        
        content_loss_value = content_loss(content_features, gen_image_content_features)
        style_loss_value = style_loss(style_features, gen_image_style_features)
        total_loss_value = total_loss(style_loss_value, content_loss_value, style_weight, content_weight)

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    gradients = tape.gradient(total_loss_value, gen_image)
    
    optimizer.apply_gradients([(gradients, gen_image)])
    
    gen_image.assign(clip(gen_image, min_vals, max_vals))
    return total_loss_value, style_loss_value, content_loss_value


def style_transfer(model, style_path, content_path, learning_rate, style_weight, content_weight, epochs, save_path):
    history = {"content_loss": [], "style_loss": [], "total_loss": []}
    
    style_image = img_preprocess(style_path)
    content_image = img_preprocess(content_path)

    style_features = get_features(model, style_image)["style"]
    content_features = get_features(model, content_image)["content"]

    content_img_shape = tf.shape(content_image)
    noise = tf.random.uniform(content_img_shape, minval=0, maxval=0.5)
    gen_image = tf.add(content_image, noise)
    gen_image = tf.Variable(gen_image, tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.99, epsilon=1e-8)
    
    print("Starting Style Transfer...")
    
    for i in range(epochs+1):
        all_losses = gradient_descent(
            model,
            optimizer,
            gen_image,
            style_features,
            content_features,
            style_weight,
            content_weight
        )

        history["total_loss"].append(all_losses[0].numpy())
        history["style_loss"].append(all_losses[1].numpy())
        history["content_loss"].append(all_losses[2].numpy())


        
        if i % 100 == 0:
            display.clear_output(wait=True)

            print("=" * 50)
            print(f"Epoch: {i}/{epochs}")
            print(f"Total Loss: {all_losses[0].numpy():.4f}")
            print(f"Style Loss: {all_losses[1].numpy():.4f}")
            print(f"Content Loss: {all_losses[2].numpy():.4f}")
            print("=" * 50)

    final_image = PIL.Image.fromarray(deprocess_img(gen_image.numpy()))
    final_image.save(save_path)
                     
    print("Style Transfer Completed!")
    print(f"Final image saved as {save_path}")

    return history







if __name__ == "__main__":
    print("=" * 50)
    check_environment()
    print("=" * 50)


    default_learning_rate = 5.0
    default_style_weight = 10
    default_content_weight = 1e3
    default_epochs = 3000


    if len(sys.argv) < 4 or len(sys.argv) > 8:
        print("Usage: python train.py <content_image_path> <style_image_path> <output_image_path> [learning_rate] [style_weight] [content_weight] [epochs]")
        sys.exit(1)

    content_image_path = sys.argv[1]
    style_image_path = sys.argv[2]
    output_image_path = sys.argv[3]
    

    learning_rate = default_learning_rate
    style_weight = default_style_weight
    content_weight = default_content_weight
    epochs = default_epochs

    if len(sys.argv) > 4:
        learning_rate = float(sys.argv[4])
    if len(sys.argv) > 5:
        content_weight = float(sys.argv[5])
    if len(sys.argv) > 6:
        style_weight = float(sys.argv[6])
    if len(sys.argv) > 7:
        epochs = int(sys.argv[7])

    arguments_dict = {
        "model": transfer_model,
        "style_path": style_image_path,
        "content_path": content_image_path,
        "learning_rate": learning_rate,
        "style_weight": style_weight,
        "content_weight": content_weight,
        "epochs": epochs,
        "save_path": output_image_path
    }

    history = style_transfer(**arguments_dict)