import unittest
from app import app, db
from app import User, Course, Assignment, Submission
import os
import tempfile

class NLPAssignmentTrackerTests(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        with app.app_context():
            db.create_all()
            # Create a test instructor
            instructor = User(username='test_instructor', email='instructor@test.com', is_instructor=True)
            instructor.set_password('password')
            db.session.add(instructor)
            
            # Create a test student
            student = User(username='test_student', email='student@test.com', is_instructor=False)
            student.set_password('password')
            db.session.add(student)
            
            # Create a test course
            course = Course(name='Test Course', code='TEST101', description='Test course description', instructor_id=1)
            db.session.add(course)
            
            # Create a test assignment
            from datetime import datetime, timedelta
            due_date = datetime.utcnow() + timedelta(days=7)
            assignment = Assignment(
                title='Test Assignment',
                description='This is a test assignment description',
                due_date=due_date,
                course_id=1,
                instructor_id=1
            )
            db.session.add(assignment)
            db.session.commit()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_index_page(self):
        response = self.app.get('/', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'NLP-Powered Assignment Tracking System', response.data)

    def test_login(self):
        response = self.app.post('/login', data=dict(
            username='test_instructor',
            password='password'
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)

    def test_login_invalid(self):
        response = self.app.post('/login', data=dict(
            username='test_instructor',
            password='wrong_password'
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid username or password', response.data)

    def test_instructor_dashboard(self):
        # Login as instructor
        response = self.app.post('/login', data=dict(
            username='test_instructor',
            password='password'
        ), follow_redirects=True)
        
        # Access instructor dashboard
        response = self.app.get('/instructor/dashboard', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test Assignment', response.data)

    def test_student_dashboard(self):
        # Login as student
        response = self.app.post('/login', data=dict(
            username='test_student',
            password='password'
        ), follow_redirects=True)
        
        # Access student dashboard
        response = self.app.get('/student/dashboard', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test Assignment', response.data)

    def test_nlp_keyword_extraction(self):
        from app import extract_keywords
        
        text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence
        concerned with the interactions between computers and human language, in particular how to program computers
        to process and analyze large amounts of natural language data. The goal is a computer capable of understanding
        the contents of documents.
        """
        
        keywords = extract_keywords(text, n=5)
        self.assertEqual(len(keywords), 5)
        
        # Check if 'language' is one of the top keywords
        keyword_terms = [k[0] for k in keywords]
        self.assertTrue('language' in keyword_terms or 'natural' in keyword_terms or 'processing' in keyword_terms)

    def test_similarity_calculation(self):
        from app import calculate_similarity
        
        text1 = "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
        text2 = "Machine learning is a type of artificial intelligence that allows software applications to become more accurate in predicting outcomes."
        
        similarity = calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.3)  # Texts should have some similarity
        
        text3 = "The capital of France is Paris."
        similarity = calculate_similarity(text1, text3)
        self.assertLess(similarity, 0.3)  # These texts should have low similarity

if __name__ == '__main__':
    unittest.main() 