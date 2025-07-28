# Welcome to UT Health San Antonio STEM STAIRWAY 2025

## About This Workshop

Welcome to **STEM STAIRWAY** hosted by UT Health San Antonio! This intensive program is designed to introduce educators to the fundamentals of programming and artificial intelligence. This will set everything up for your use of vibe coding
## Workshop Goals

The primary goal of this module is to **familiarize you with basic programming principles** and introduce you to the world of machine learning. By the end of this workshop, you will:

- Understand fundamental programming concepts in Python
- Learn object-oriented programming principles
- Explore machine learning basics through hands-on projects
- Vibe Code Full Applications
## Part 1 Workshop Structure

This part is organized into two main modules:

### 1. Introduction to Python (`Introduction_to_python/`)
- **Lesson 1**: Functions and Basic Programming Concepts
- **Jupyter Notebook**: DataTypes and Control FLow

### 2. Introduction to Machine Learning (`Introduction_to_machine_learning/`)
- **Gradient Descent with Mean Absolute Error (MAE)**
- **Linear Regression from Scratch**
- **Hands-on Experiments and Practice Exercises**
- **Data Visualization and Analysis**

## Getting Started

### Prerequisites

Before starting the workshop, you'll need:
- A computer with Python 3.8 or higher
- Basic familiarity with using a command line/terminal
- Curiosity and enthusiasm for learning!

### Setting Up Your Environment

#### Step 1: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated from other Python projects on your system.

**On macOS/Linux:**
```bash
# Navigate to the workshop directory
cd STEM_STAIRWAY_2025_PART_1

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Navigate to the workshop directory
cd STEM_STAIRWAY_2025_PART_1

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

#### Step 2: Install Required Packages

Once your virtual environment is activated, install the required packages:

```bash
# Install all required packages
pip install -r requirements.txt
```

#### Step 3: Verify Installation

Test that everything is working correctly:

```bash
# Test Python installation
python --version

# Test key packages
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
python -c "import jupyter; print('Jupyter version:', jupyter.__version__)"
```

### Deactivating the Virtual Environment

When you're done working on the workshop, you can deactivate the virtual environment:

```bash
deactivate
```

## How to Use This Workshop

### 1. Start with Python Fundamentals

Begin your journey in the `Introduction_to_python/` directory:

```bash
# Navigate to Python module
cd Introduction_to_python

# Start with the Jupyter notebook for interactive learning
jupyter notebook jupyter_lesson.ipynb
```

### 2. Progress Through the Lessons

Each lesson builds upon the previous one:

- **Lesson 1**: Functions - Learn to write reusable code
- **Jupyter Notebook** - DataTypes and Control FLow

### 3. Explore Machine Learning

Once comfortable with Python, move to machine learning:

```bash
# Navigate to ML module
cd Introduction_to_machine_learning

# Run the complete lesson
python lesson.py

# Complete practice exercises
python practice.py
```