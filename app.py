import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import nltk
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///nlp_assignment_tracker.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # If model isn't downloaded yet
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_instructor = db.Column(db.Boolean, default=False)
    assignments = db.relationship('Assignment', backref='instructor', lazy=True)
    submissions = db.relationship('Submission', backref='student', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(10), nullable=False, unique=True)
    description = db.Column(db.Text)
    instructor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assignments = db.relationship('Assignment', backref='course', lazy=True)

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    due_date = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    instructor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    submissions = db.relationship('Submission', backref='assignment', lazy=True)
    keywords = db.relationship('AssignmentKeyword', backref='assignment', lazy=True)

class AssignmentKeyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    keyword = db.Column(db.String(50), nullable=False)
    weight = db.Column(db.Float, default=1.0)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)
    feedback = db.Column(db.Text)
    grade = db.Column(db.Float)
    similarity_score = db.Column(db.Float)
    keywords = db.relationship('SubmissionKeyword', backref='submission', lazy=True)

class SubmissionKeyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    keyword = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, default=0.0)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)

# NLP Helper Functions
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def extract_keywords(text, n=10):
    """Extract important keywords using TF-IDF"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    
    # Get feature names and TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Create a dictionary of words and their TF-IDF scores
    word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
    
    # Sort words by TF-IDF score and get top n
    top_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    return top_keywords

def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts"""
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity

def generate_feedback(submission_text, assignment_text, assignment_keywords=[]):
    """Generate automated feedback for a submission"""
    # Extract keywords if not provided
    if not assignment_keywords:
        assignment_keywords = [k[0] for k in extract_keywords(assignment_text)]
    
    # Extract submission keywords
    submission_keywords = [k[0] for k in extract_keywords(submission_text)]
    
    # Calculate similarity
    similarity = calculate_similarity(assignment_text, submission_text)
    
    # Check which assignment keywords are missing in the submission
    missing_keywords = [k for k in assignment_keywords if k not in submission_keywords]
    
    # Generate feedback text
    feedback_text = f"Similarity Score: {similarity:.2f}\n\n"
    
    if similarity > 0.7:
        feedback_text += "Great job! Your submission is highly relevant to the assignment.\n\n"
    elif similarity > 0.4:
        feedback_text += "Good effort. Your submission addresses some key aspects of the assignment.\n\n"
    else:
        feedback_text += "Your submission may not fully address the assignment requirements. Please review the instructions.\n\n"
    
    if missing_keywords:
        feedback_text += f"Consider addressing these important concepts: {', '.join(missing_keywords)}.\n\n"
    
    # Add specific feedback based on content analysis
    doc = nlp(submission_text)
    
    # Check for readability
    sentences = list(doc.sents)
    avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0
    
    if avg_sentence_length > 40:
        feedback_text += "Consider using shorter sentences to improve readability.\n\n"
    
    # Check for passive voice (simplified approach)
    passive_count = len([sent for sent in sentences if " was " in sent.text.lower() or " were " in sent.text.lower() or " been " in sent.text.lower()])
    if passive_count / len(sentences) > 0.3 if sentences else 0:
        feedback_text += "Consider using more active voice in your writing.\n\n"
    
    return feedback_text, similarity

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.is_instructor:
            return redirect(url_for('instructor_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user is None or not user.check_password(password):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        is_instructor = 'instructor' in request.form
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email, is_instructor=is_instructor)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if current_user.is_instructor:
        return redirect(url_for('instructor_dashboard'))
    
    # Get assignments for courses the student is enrolled in
    # This would require a student-course enrollment table, simplified here
    assignments = Assignment.query.all()
    submissions = Submission.query.filter_by(student_id=current_user.id).all()
    
    return render_template('student_dashboard.html', assignments=assignments, submissions=submissions)

@app.route('/instructor/dashboard')
@login_required
def instructor_dashboard():
    if not current_user.is_instructor:
        return redirect(url_for('student_dashboard'))
    
    courses = Course.query.filter_by(instructor_id=current_user.id).all()
    assignments = Assignment.query.filter_by(instructor_id=current_user.id).all()
    
    return render_template('instructor_dashboard.html', courses=courses, assignments=assignments)

@app.route('/assignments/<int:assignment_id>')
@login_required
def view_assignment(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)
    
    if current_user.is_instructor:
        submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
        return render_template('assignment_instructor_view.html', assignment=assignment, submissions=submissions)
    else:
        submission = Submission.query.filter_by(assignment_id=assignment_id, student_id=current_user.id).first()
        return render_template('assignment_student_view.html', assignment=assignment, submission=submission)

@app.route('/assignments/<int:assignment_id>/submit', methods=['GET', 'POST'])
@login_required
def submit_assignment(assignment_id):
    if current_user.is_instructor:
        return redirect(url_for('view_assignment', assignment_id=assignment_id))
    
    assignment = Assignment.query.get_or_404(assignment_id)
    
    # Check if assignment is past due date
    if datetime.utcnow() > assignment.due_date:
        flash('This assignment is past the due date.')
        return redirect(url_for('view_assignment', assignment_id=assignment_id))
    
    # Check if already submitted
    existing_submission = Submission.query.filter_by(assignment_id=assignment_id, student_id=current_user.id).first()
    
    if request.method == 'POST':
        content = request.form.get('content')
        
        # Generate NLP analysis
        feedback_text, similarity_score = generate_feedback(content, assignment.description)
        
        if existing_submission:
            # Update existing submission
            existing_submission.content = content
            existing_submission.submitted_at = datetime.utcnow()
            existing_submission.similarity_score = similarity_score
            existing_submission.feedback = feedback_text
            
            # Clear old keywords
            SubmissionKeyword.query.filter_by(submission_id=existing_submission.id).delete()
            
            # Save new keywords
            for keyword, score in extract_keywords(content):
                keyword_entry = SubmissionKeyword(
                    keyword=keyword,
                    score=score,
                    submission_id=existing_submission.id
                )
                db.session.add(keyword_entry)
            
            db.session.commit()
            flash('Submission updated!')
        else:
            # Create new submission
            submission = Submission(
                content=content,
                student_id=current_user.id,
                assignment_id=assignment_id,
                similarity_score=similarity_score,
                feedback=feedback_text
            )
            db.session.add(submission)
            db.session.commit()
            
            # Save keywords
            for keyword, score in extract_keywords(content):
                keyword_entry = SubmissionKeyword(
                    keyword=keyword,
                    score=score,
                    submission_id=submission.id
                )
                db.session.add(keyword_entry)
            
            db.session.commit()
            flash('Assignment submitted!')
        
        return redirect(url_for('view_assignment', assignment_id=assignment_id))
    
    return render_template('submit_assignment.html', assignment=assignment, submission=existing_submission)

@app.route('/instructor/assignments/create', methods=['GET', 'POST'])
@login_required
def create_assignment():
    if not current_user.is_instructor:
        return redirect(url_for('student_dashboard'))
    
    courses = Course.query.filter_by(instructor_id=current_user.id).all()
    
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        due_date_str = request.form.get('due_date')
        course_id = request.form.get('course_id')
        
        # Parse due date
        due_date = datetime.strptime(due_date_str, '%Y-%m-%dT%H:%M')
        
        # Create assignment
        assignment = Assignment(
            title=title,
            description=description,
            due_date=due_date,
            course_id=course_id,
            instructor_id=current_user.id
        )
        db.session.add(assignment)
        db.session.commit()
        
        # Extract and save keywords
        for keyword, score in extract_keywords(description):
            keyword_entry = AssignmentKeyword(
                keyword=keyword,
                weight=score,
                assignment_id=assignment.id
            )
            db.session.add(keyword_entry)
        
        db.session.commit()
        flash('Assignment created!')
        return redirect(url_for('instructor_dashboard'))
    
    return render_template('create_assignment.html', courses=courses)

@app.route('/submissions/<int:submission_id>/grade', methods=['POST'])
@login_required
def grade_submission(submission_id):
    if not current_user.is_instructor:
        return redirect(url_for('student_dashboard'))
    
    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.get(submission.assignment_id)
    
    if assignment.instructor_id != current_user.id:
        flash('You do not have permission to grade this submission.')
        return redirect(url_for('instructor_dashboard'))
    
    grade = request.form.get('grade')
    feedback = request.form.get('feedback')
    
    submission.grade = float(grade)
    submission.feedback = feedback
    db.session.commit()
    
    flash('Submission graded!')
    return redirect(url_for('view_assignment', assignment_id=submission.assignment_id))

@app.route('/courses/create', methods=['GET', 'POST'])
@login_required
def create_course():
    if not current_user.is_instructor:
        return redirect(url_for('student_dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        code = request.form.get('code')
        description = request.form.get('description')
        
        course = Course(
            name=name,
            code=code,
            description=description,
            instructor_id=current_user.id
        )
        db.session.add(course)
        db.session.commit()
        
        flash('Course created!')
        return redirect(url_for('instructor_dashboard'))
    
    return render_template('create_course.html')

@app.route('/api/analyze-text', methods=['POST'])
@login_required
def analyze_text():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Extract keywords
    keywords = extract_keywords(text)
    
    # Analyze readability
    doc = nlp(text)
    sentences = list(doc.sents)
    avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0
    
    # Entity recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return jsonify({
        'keywords': [{'keyword': k, 'score': float(s)} for k, s in keywords],
        'readability': {
            'avg_sentence_length': float(avg_sentence_length),
            'sentence_count': len(sentences)
        },
        'entities': [{'text': e, 'type': t} for e, t in entities]
    })

# Initialize the database
with app.app_context():
    db.create_all()
    
    # Create admin user if none exists
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@example.com',
            is_instructor=True
        )
        admin.set_password('admin')
        db.session.add(admin)
        
        # Create a test student
        student = User(
            username='student',
            email='student@example.com',
            is_instructor=False
        )
        student.set_password('student')
        db.session.add(student)
        
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True) 