from PIL import Image
import numpy as np

def add_noise_to_image(image_path, output_path, mean=0, stddev=25):
    """
    Adds noise to an image and saves it.

    :param image_path: Path to the input image.
    :param output_path: Path to save the noisy image.
    :param mean: Mean of the noise distribution (default is 0).
    :param stddev: Standard deviation of the noise distribution (default is 25).
    """
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Generate random noise
    noise = np.random.normal(mean, stddev, image_array.shape)

    # Add noise to the image
    noisy_image_array = image_array + noise

    # Ensure values are in valid range (0-255)
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)

    # Convert back to an image
    noisy_image = Image.fromarray(noisy_image_array)

    # Save the noisy image
    noisy_image.save(output_path)
    print(f"Noisy image saved as {output_path}")

# Example usage
input_image = "bmw.png"
output_image = "noisy_image.png"
mean = 0          # Adjust mean of noise
stddev = 0       # Adjust standard deviation of noise

add_noise_to_image(input_image, output_image, mean, stddev)
