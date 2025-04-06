from flask import Flask, render_template, request, redirect, url_for, session
import os
import sqlite3
import fitz  # PyMuPDF
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and vectorizer
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def init_db():
    conn = sqlite3.connect('database/candidates.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS candidates (
                    name TEXT,
                    score REAL
                )''')
    conn.commit()
    conn.close()

def save_job_to_db(jd_text):
    conn = sqlite3.connect('database/candidates.db')
    c = conn.cursor()
    c.execute("INSERT INTO jobs (description) VALUES (?)", (jd_text,))
    conn.commit()
    conn.close()

def save_to_db(data):
    conn = sqlite3.connect('database/candidates.db')
    c = conn.cursor()
    c.execute('DELETE FROM candidates')  # Clear old entries
    c.executemany('INSERT INTO candidates (name, score) VALUES (?, ?)', data)
    conn.commit()
    conn.close()

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/choose')
def choose():
    return render_template('choose.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database/candidates.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login', registered=True))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username already exists.")

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database/candidates.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = username
            return redirect(url_for('admin'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html', registered=request.args.get('registered'))

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        jd_text = request.form['jd']
        uploaded_files = request.files.getlist('resumes')
        results = []

        save_job_to_db(jd_text)

        if not uploaded_files or uploaded_files[0].filename == '':
            return render_template('thank_you.html')

        for file in uploaded_files:
            filename = file.filename
            if filename.endswith('.pdf'):
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)

                resume_text = extract_text_from_pdf(path)
                if len(resume_text.strip()) < 100 or len(jd_text.strip()) < 20:
                    prediction = 0.0
                else:
                    combined_text = jd_text + " " + resume_text
                    vec = vectorizer.transform([combined_text])
                    prediction = model.predict_proba(vec)[0][1]

                results.append((filename, round(prediction, 2)))

        results.sort(key=lambda x: x[1], reverse=True)
        save_to_db(results)
        return redirect(url_for('shortlisted'))

    return render_template('admin.html')

@app.route('/shortlisted')
def shortlisted():
    conn = sqlite3.connect('database/candidates.db')
    c = conn.cursor()
    c.execute("SELECT name, score FROM candidates WHERE score > 0.5 ORDER BY score DESC")
    data = c.fetchall()
    conn.close()
    return render_template('shortlisted.html', candidates=data)

@app.route('/jobs')
def jobs():
    conn = sqlite3.connect('database/candidates.db')
    c = conn.cursor()
    c.execute("SELECT id, description FROM jobs ORDER BY id DESC")
    job_list = c.fetchall()
    conn.close()
    return render_template('jobs.html', jobs=job_list)

@app.route('/apply/<int:job_id>', methods=['GET', 'POST'])
def apply(job_id):
    if request.method == 'POST':
        uploaded_file = request.files.get('resume')
        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(path)

            resume_text = extract_text_from_pdf(path)

            conn = sqlite3.connect('database/candidates.db')
            c = conn.cursor()
            c.execute("SELECT description FROM jobs WHERE id = ?", (job_id,))
            row = c.fetchone()
            conn.close()

            if row:
                jd_text = row[0]
                combined_text = jd_text + " " + resume_text
                vec = vectorizer.transform([combined_text])
                prediction = model.predict_proba(vec)[0][1]
                score = round(prediction * 100, 2)

                conn = sqlite3.connect('database/candidates.db')
                c = conn.cursor()
                c.execute("INSERT INTO candidates (name, score) VALUES (?, ?)", (uploaded_file.filename, score / 100))
                conn.commit()
                conn.close()

            return render_template('apply_success.html')

    return render_template('apply.html', job_id=job_id)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
