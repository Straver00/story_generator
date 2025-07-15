from story_generator import (
    generate_story_from_natural_language, 
    generate_image_from_prompt,
    init_client
)
from dotenv import load_dotenv
import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

# Carga de variables de entorno desde un archivo .env
try:
    load_dotenv()
except ImportError:
    print("Por favor, instala el paquete 'python-dotenv' para cargar las variables de entorno desde un archivo .env.")
    load_dotenv = None
except Exception as e:
    print(f"Se produjo un error al cargar las variables de entorno: {e}")
    load_dotenv = None

api_key_history = os.getenv("GOOGLE_API_KEY")
api_key_image = os.getenv("IMAGE_API_KEY")

# Creamos los clientes para cada solicitud
client_history = init_client(api_key_history)
client_image = init_client(api_key_image)

# Generación de una historia a partir de un prompt en lenguaje natural
history, prompt_image = generate_story_from_natural_language(
    client_history,
    user_prompt="Escribe una historia de ciencia ficción sobre un viaje a Marte.",
)

# Generamos la imagen a partir del prompt generado
image = generate_image_from_prompt(client_image, prompt_image)

# Guardamos la imagen en un archivo con scikit-image
if image is not None:
    skimage.io.imsave("generated_image.png", image)
    print("Imagen guardada como 'generated_image.png'.")
else:
    print("No se pudo generar la imagen. Verifique el prompt o la configuración del modelo.")

print("Historia e imagen generadas con éxito.")
