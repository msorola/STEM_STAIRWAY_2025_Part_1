"""
PRACTICE: Functions Fundamentals
===============================

This is your practice file for functions fundamentals.

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
    pass

def calculate_grade_stats(grades: list) -> dict:
    """
    Calculate statistics for a list of grades.

    Args:
        grades: List of numeric grades

    Returns:
        Dictionary containing:
        - average: Average grade
        - highest: Highest grade
        - lowest: Lowest grade
        - passing: Number of passing grades (>= 70)

    Example:
        >>> calculate_grade_stats([85, 92, 78, 65, 88])
        {'average': 81.6, 'highest': 92, 'lowest': 65, 'passing': 4}
    """
    pass

def format_name(first: str, last: str, middle: str = "") -> str:
    """
    Format a person's name with proper capitalization.

    Args:
        first: First name
        last: Last name
        middle: Middle name (optional)

    Returns:
        Properly formatted full name

    Example:
        >>> format_name("john", "doe")
        'John Doe'
        >>> format_name("mary", "jane", "ann")
        'Mary Ann Jane'
    """
    pass

def calculate_final_grade(assignments: list, exams: list, weights: tuple = (0.4, 0.6)) -> float:
    """
    Calculate final grade based on assignments and exams with weights.

    Args:
        assignments: List of assignment scores
        exams: List of exam scores
        weights: Tuple of (assignment_weight, exam_weight), defaults to (0.4, 0.6)

    Returns:
        Weighted final grade rounded to 1 decimal place

    Example:
        >>> calculate_final_grade([85, 90, 88], [78, 85], (0.4, 0.6))
        84.0
    """
    pass

def get_letter_grade(score: float) -> str:
    """
    Convert a numeric score to a letter grade.

    Args:
        score: Numeric score

    Returns:
        Letter grade (A, B, C, D, or F)

    Example:
        >>> get_letter_grade(85)
        'B'
        >>> get_letter_grade(92)
        'A'
    """
    pass

# === TESTING ===
if __name__ == "__main__":
    print("=== FUNCTION PRACTICE TESTS ===\n")

    # Test get_top_students
    print("Testing get_top_students...")
    student_grades = {
        'Alice': 85,
        'Bob': 72,
        'Charlie': 90,
        'Diana': 65
    }
    passing_score = 80
    result = get_top_students(student_grades, passing_score)
    assert sorted(result) == ['Alice', 'Charlie'], f"Expected ['Alice', 'Charlie'] but got {result}"
    print("✓ get_top_students passed")

    # Test calculate_grade_stats
    print("\nTesting calculate_grade_stats...")
    grades = [85, 92, 78, 65, 88]
    stats = calculate_grade_stats(grades)
    assert stats['average'] == 81.6, f"Expected average 81.6 but got {stats['average']}"
    assert stats['highest'] == 92, f"Expected highest 92 but got {stats['highest']}"
    assert stats['passing'] == 4, f"Expected 4 passing grades but got {stats['passing']}"
    print("✓ calculate_grade_stats passed")

    # Test format_name
    print("\nTesting format_name...")
    assert format_name("john", "doe") == "John Doe"
    assert format_name("mary", "jane", "ann") == "Mary Ann Jane"
    print("✓ format_name passed")

    # Test calculate_final_grade
    print("\nTesting calculate_final_grade...")
    assignments = [85, 90, 88]
    exams = [78, 85]
    final_grade = calculate_final_grade(assignments, exams)
    assert final_grade == 84.0, f"Expected 84.2 but got {final_grade}"
    print("✓ calculate_final_grade passed")

    # Test get_letter_grade
    print("\nTesting get_letter_grade...")
    assert get_letter_grade(85) == 'B'
    assert get_letter_grade(92) == 'A'
    assert get_letter_grade(75) == 'C'
    assert get_letter_grade(65) == 'D'
    assert get_letter_grade(55) == 'F'
    print("✓ get_letter_grade passed")

    print("\nAll tests passed successfully! 🎉")
