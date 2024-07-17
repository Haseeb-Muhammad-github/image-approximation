import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def svd_image_approximation(image_path, k):
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(image_array, full_matrices=False)
    
    # Keep only the top k singular values
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct the approximated image
    approximated_image_array = np.dot(U_k, np.dot(S_k, Vt_k))
    
    # Clip values to be in the valid range [0, 255] and convert to uint8
    approximated_image_array = np.clip(approximated_image_array, 0, 255).astype(np.uint8)
    
    # Convert array back to image
    approximated_image = Image.fromarray(approximated_image_array)
    
    return image, approximated_image

# Path to your image
image_path = 'your-image.jpeg'

# Number of singular values to keep (rank of approximation)
k = 50

# Perform SVD image approximation
original_image, approximated_image = svd_image_approximation(image_path, k)

# Display the original and approximated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Approximated Image (k={k})')
plt.imshow(approximated_image, cmap='gray')
plt.axis('off')

plt.show()

# Save the approximated image
approximated_image.save('flower_image.jpeg')
