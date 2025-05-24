import os
import torch
import tensorflow as tf
import sqlite3
from datetime import datetime
from flask import Flask, request, render_template, url_for, redirect, session, flash
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
from torch_model import BreastCancerCNN  # Assure-toi que cette classe correspond bien à ton modèle

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "clé_secrète_de_dev")

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Chargement modèle PyTorch ---
torch_model = BreastCancerCNN()

try:
    pretrained_weights = torch.load("resnet18_breast_cancer.pth", map_location=torch.device('cpu'))
    model_dict = torch_model.state_dict()
    compatible_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(compatible_weights)
    torch_model.load_state_dict(model_dict)
    torch_model.eval()
    print("Modèle PyTorch chargé avec succès.")
except Exception as e:
    print("Erreur lors du chargement du modèle PyTorch :", e)

# --- Chargement modèle TensorFlow ---
try:
    tf_model = tf.keras.models.load_model("rokhaya_model.tensorflow")
    print("Modèle TensorFlow chargé avec succès.")
except Exception as e:
    print("Erreur lors du chargement du modèle TensorFlow :", e)

# --- Transformations PyTorch avec normalisation ImageNet standard ---
torch_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # valeurs classiques ImageNet
                         std=[0.229, 0.224, 0.225])
])

def init_db():
    with sqlite3.connect('database.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                image_filename TEXT,
                model_used TEXT,
                prediction_result INTEGER,
                prediction_date TEXT
            )
        ''')
        conn.commit()

        c.execute("SELECT * FROM users WHERE username = ?", ('medecin1',))
        if c.fetchone() is None:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('medecin1', 'motdepasse1'))
            conn.commit()

init_db()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_torch = torch_transform(image).unsqueeze(0)  # Shape [1, 3, 224, 224]
    
    # Pour TF : conversion en array + redimension + normalisation [0,1]
    img_tf = image.resize((224, 224))
    img_tf = tf.keras.preprocessing.image.img_to_array(img_tf) / 255.0
    img_tf = tf.expand_dims(img_tf, axis=0)  # Shape [1, 224, 224, 3]
    
    return img_torch, img_tf

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect('database.db') as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
            user = c.fetchone()

        if user:
            session['username'] = username
            flash(f"Bienvenue {username} !", "success")
            return redirect(url_for('index'))
        else:
            flash("Nom d'utilisateur ou mot de passe incorrect.", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Vous avez été déconnecté.", "info")
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    prediction = ''
    img_filename = ''

    if request.method == 'POST':
        try:
            model_type = request.form['model']
            patient_name = request.form['patient_name']
            file = request.files['image']

            if not (file and allowed_file(file.filename)):
                flash("Erreur : format de fichier non autorisé.", "danger")
                return render_template('index.html', prediction=prediction, img=img_filename)

            img_filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            file.save(path)

            img_torch, img_tf = preprocess_image(path)

            class_names = ['pituitary', 'notumor','meningioma','glioma']

            if model_type == 'pytorch':
                with torch.no_grad():
                    outputs = torch_model(img_torch)  # logits attendus
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred].item()
                print(f"PyTorch Prediction: {class_names[pred]} (confidence: {confidence:.2f})")
            elif model_type == 'tensorflow':
                outputs = tf_model.predict(img_tf)  # probabilités softmax attendues
                pred = tf.argmax(outputs, axis=1).numpy()[0]
                confidence = outputs[0][pred]
                print(f"TensorFlow Prediction: {class_names[pred]} (confidence: {confidence:.2f})")
            else:
                raise ValueError("Modèle inconnu.")

            disease_label = 0 if class_names[pred] == 'notumor' else 1
            prediction = f"Tumeur détectée : {disease_label} ({class_names[pred]}) - confiance : {confidence:.2f}"

            # Sauvegarde dans la base
            with sqlite3.connect('database.db') as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO predictions (patient_name, image_filename, model_used, prediction_result, prediction_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (patient_name, img_filename, model_type, pred, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()

        except Exception as e:
            prediction = f"Erreur : {str(e)}"
            flash(prediction, "danger")

    return render_template('index.html', prediction=prediction, img=img_filename)

@app.route('/historique')
def historique():
    if 'username' not in session:
        return redirect(url_for('login'))

    with sqlite3.connect('database.db') as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY prediction_date DESC')
        data = c.fetchall()

    return render_template('historique.html', data=data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)








