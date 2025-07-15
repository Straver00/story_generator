# app.py
from flask import Flask, render_template, request
import json
import base64
import os
from io import BytesIO
import numpy as np
from dotenv import load_dotenv
from story_generator import (
    generate_story_from_natural_language,
    generate_image_from_prompt,
    init_client
)

# Carga variables de entorno desde .env si existe
try:
    load_dotenv()
except Exception:
    pass  # Continuar aunque dotenv no esté instalado

# Claves de API
api_key_history = os.getenv("GOOGLE_API_KEY")
api_key_image = os.getenv("IMAGE_API_KEY")
if not api_key_history or not api_key_image:
    raise RuntimeError("Define GOOGLE_API_KEY e IMAGE_API_KEY en tus variables de entorno.")

# Inicializa clientes
client_history = init_client(api_key_history)
client_image = init_client(api_key_image)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    story = None
    img_data = None
    error = None
    if request.method == 'POST':
        mode = request.form.get('mode')
        try:
            if mode == 'chat':
                user_prompt = request.form.get('chat_input', '').strip()
                # Llamada al generador de historia y prompt de imagen
                story, prompt_image = generate_story_from_natural_language(
                    client_history,
                    user_prompt=user_prompt
                )
            else:
                attributes = {
                    'personajes': request.form.get('personajes'),
                    'genero': request.form.get('genero'),
                    'escenario': request.form.get('escenario'),
                    'tono': request.form.get('tono'),
                    'elementos_trama': request.form.get('elementos_trama'),
                    'longitud': request.form.get('longitud')
                }
                # Construye un prompt natural basado en JSON
                user_prompt = f"Genera una historia {attributes['longitud']} de género {attributes['genero']} " \
                              f"con personajes {attributes['personajes']}, ambientada en {attributes['escenario']}, " \
                              f"con tono {attributes['tono']} y elementos de trama {attributes['elementos_trama']}."
                story, prompt_image = generate_story_from_natural_language(
                    client_history,
                    user_prompt=user_prompt
                )
            # Generar imagen usando el prompt de imagen devuelto
            image_array = generate_image_from_prompt(client_image, prompt_image)
            if image_array is not None:
                # Codificar imagen a base64 para HTML
                buffered = BytesIO()
                from PIL import Image
                Image.fromarray(image_array.astype('uint8')).save(buffered, format='PNG')
                img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            error = str(e)
    return render_template('index.html', story=story, img_data=img_data, error=error)

if __name__ == '__main__':
    # Ejecutar Flask en modo debug
    app.run(debug=True)
