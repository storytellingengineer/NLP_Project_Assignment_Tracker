# NLP Assignment Tracking System

An intelligent web-based platform that leverages Natural Language Processing (NLP) to enhance the educational assignment experience for both students and instructors. This system provides automated analysis, feedback, and tracking for academic assignments.

## Features

- **NLP-Powered Assignment Analysis**: Automatically extract keywords, topics, and concepts from assignments and submissions
- **Intelligent Feedback Generation**: Provide instant, tailored feedback to students based on NLP analysis
- **Similarity Scoring**: Measure the relevance of submissions to assignment requirements
- **Concept Coverage Detection**: Identify missing concepts in student submissions
- **Readability Analysis**: Assess and provide feedback on writing quality
- **Dashboard Analytics**: Visualize performance metrics and learning progress
- **User Roles**: Separate interfaces for students and instructors

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Database**: SQLite (development), PostgreSQL (production)
- **NLP Libraries**: NLTK, SpaCy, scikit-learn
- **Machine Learning**: Transformers, Hugging Face models
- **Containerization**: Docker

## Installation

### Prerequisites

- Python 3.10+
- Docker (optional)
- Git

### Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nlp-assignment-tracker.git
   cd nlp-assignment-tracker
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download required NLP models:
   ```
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords wordnet
   ```

5. Set environment variables (create a `.env` file):
   ```
   SECRET_KEY=your-secret-key
   DATABASE_URL=sqlite:///nlp_assignment_tracker.db
   FLASK_APP=app.py
   FLASK_DEBUG=1
   ```

6. Initialize the database:
   ```
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

7. Run the application:
   ```
   flask run
   ```

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t nlp-assignment-tracker .
   ```

2. Run the container:
   ```
   docker run -d -p 5000:5000 --name assignment-tracker nlp-assignment-tracker
   ```

## Usage

### For Instructors

1. Register an account with the "Instructor" role
2. Create courses and assignments
3. Monitor student submissions
4. Review automated analysis and feedback
5. Add manual feedback and grades as needed

### For Students

1. Register a student account
2. Browse available assignments
3. Submit work for analysis
4. Receive instant NLP-powered feedback
5. Track progress and performance metrics

## NLP Components

### Keyword Extraction
The system uses TF-IDF (Term Frequency-Inverse Document Frequency) to identify the most important terms in both assignments and submissions.

### Similarity Analysis
Cosine similarity measures how closely a submission aligns with assignment requirements, helping to assess relevance.

### Topic Modeling
Latent Dirichlet Allocation (LDA) identifies underlying topics in text, useful for discovering themes in both assignments and submissions.

### Sentiment and Readability Analysis
The system analyzes writing style, sentence structure, and complexity to provide feedback on communication skills.

## Project Structure

```
nlp_assignment_tracker/
├── app.py                 # Main Flask application
├── models/                # Database models
├── nlp/                   # NLP processing components
├── static/                # Static files (CSS, JS)
├── templates/             # HTML templates
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter notebooks for model exploration
├── migrations/            # Database migrations
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by educational platforms like Google Classroom and Intellipaat LMS
- Special thanks to the open-source NLP community for their amazing tools and libraries 
