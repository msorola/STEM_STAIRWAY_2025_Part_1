"""
PRACTICE: Functions Fundamentals
===============================

This is your practice file for functions fundamentals.
Complete the TODO sections below to test your understanding.

Instructions:
1. Read the function descriptions carefully
2. Implement each function according to the specifications
3. Run this file to test your implementations
4. Use the hints if you get stuck
"""

def get_top_students(grades: dict, passing_score: int) -> list:
    """
    Return a list of students who scored above the passing_score.

    Args:
        grades: Dictionary of student names as keys and scores as values
        passing_score: Score threshold for passing

    Returns:
        A list of names of students who scored above the threshold

    Example:
        >>> get_top_students({'Alice': 85, 'Bob': 72, 'Charlie': 90}, 80)
        ['Alice', 'Charlie']
    """
    # TODO: Loop through the dictionary
    # TODO: Use if/else to check if score is above passing_score
    # TODO: Collect names of students who passed
    passed_students = []
    for name, score in grades.items():
        if score > passing_score:
            passed_students.append(name)
    return passed_students

# === TESTING ===
if __name__ == "__main__":
    print("=== IF/ELSE & LOOP PRACTICE ===\n")
    student_grades = {
        'Alice': 85,
        'Bob': 72,
        'Charlie': 90,
        'Diana': 65
    }
    passing_score = 80
    expected = ['Alice', 'Charlie']

    result = get_top_students(student_grades, passing_score)
    print("Passing score:", passing_score)
    print("Top students:", result)

    # ASSERTION TO CHECK IMPLEMENTATION
    assert sorted(result) == sorted(expected), f"Expected {expected} but got {result}"

    print("\nAll tests passed!")
