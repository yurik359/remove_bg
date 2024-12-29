from flask import Flask, request, jsonify
from endpoints.remove_bg import remove_background_bp
app = Flask(__name__)
app.register_blueprint(remove_background_bp)
print(app.url_map)

@app.route('/')
def home():
    return "Hello, Flask!"

# @app.route('/api/remove-background', methods=['POST'])
# def remove_background():
#     # Обробка запиту
#     return jsonify({"message": "Endpoint is working!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
