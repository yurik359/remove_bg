from flask import Blueprint, request, jsonify
import base64
from u2net_test import main  # Імпортуємо функцію для обробки зображення

remove_background_bp = Blueprint('remove-background', __name__)

@remove_background_bp.route('/remove-background', methods=['POST'])
def remove_background():
    try:
        
        uploaded_file = request.files['image']
        print(request)
        if not uploaded_file:
            return jsonify({'error': 'Image file is required'}), 400
      

       
        image_buffer = uploaded_file.read()

        
        result_base64 = main(image_buffer, save_dir='../test_data/u2net_results')

        return jsonify({'image': result_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
