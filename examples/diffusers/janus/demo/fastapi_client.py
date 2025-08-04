import requests
from PIL import Image
import io
# Endpoint URLs
understand_image_url = "http://localhost:8000/understand_image_and_question/"
generate_images_url = "http://localhost:8000/generate_images/"

# Use your image file path here
image_path = "images/equation.png"

# Function to call the image understanding endpoint
def understand_image_and_question(image_path, question, seed=42, top_p=0.95, temperature=0.1):
    files = {'file': open(image_path, 'rb')}
    data = {
        'question': question,
        'seed': seed,
        'top_p': top_p,
        'temperature': temperature
    }
    response = requests.post(understand_image_url, files=files, data=data)
    response_data = response.json()
    print("Image Understanding Response:", response_data['response'])


# Function to call the text-to-image generation endpoint
def generate_images(prompt, seed=None, guidance=5.0):
    data = {
        'prompt': prompt,
        'seed': seed,
        'guidance': guidance
    }
    response = requests.post(generate_images_url, data=data, stream=True)
    
    if response.ok:
        img_idx = 1

        # We will create a new BytesIO for each image
        buffers = {}

        try:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    # Use a boundary detection to determine new image start
                    if img_idx not in buffers:
                        buffers[img_idx] = io.BytesIO()

                    buffers[img_idx].write(chunk)

                    # Attempt to open the image
                    try:
                        buffer = buffers[img_idx]
                        buffer.seek(0)
                        image = Image.open(buffer)
                        img_path = f"generated_image_{img_idx}.png"
                        image.save(img_path)
                        print(f"Saved: {img_path}")

                        # Prepare the next image buffer
                        buffer.close()
                        img_idx += 1

                    except Exception as e:
                        # Continue loading data into the current buffer
                        continue

        except Exception as e:
            print("Error processing image:", e)
    else:
        print("Failed to generate images.")


# Example usage
if __name__ == "__main__":
    # Call the image understanding API
    understand_image_and_question(image_path, "What is this image about?")
    
    # Call the image generation API
    generate_images("A beautiful sunset over a mountain range, digital art.")
