"""
LESSON: Functions Fundamentals
=============================

Learning Objectives:
• Design and implement reusable functions with proper parameter handling
• Use type hints and docstrings to create self-documenting code
• Implement error handling and input validation in functions
• Understand different return patterns and function composition
"""

# === CONCEPT OVERVIEW ===
# Functions are the building blocks of reusable code in Python. They allow us to
# encapsulate logic, accept inputs (parameters), process data, and return results.
# Well-designed functions are single-purpose, have clear interfaces, and handle
# errors gracefully. This lesson focuses on creating functions that are both
# functional and maintainable.

def display_name(name: str):
    print(f"Hello, my name is {name}")
    return


def calculate_area(length: float, width: float) -> float:
    """
    Calculate the area of a rectangle given its length and width.
    
    Args:
        length: The length of the rectangle
        width: The width of the rectangle
    
    Returns:
        The area of the rectangle
        
    Raises:
        ValueError: If dimensions are negative
        
    Example:
        calculate_area(5.0, 3.0)
        15.0
    """
    #define area as length * width
    area = length * width
    #return the variable area from the function
    return area



def find_max_min(numbers: list) -> tuple:
    """
    Find the maximum and minimum values in a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
    
    Returns:
        Tuple of (maximum, minimum) values
        
    Raises:
        ValueError: If list is empty
        
    Example:
        find_max_min([1, 5, 3, 9, 2])
        (9, 1)
        find_max_min([-10, 0, 100])
        (100, -10)
    """
    #sort the list of numbers
    numbers.sort()
    lowest_number = numbers[0] #pull the first entry which should now be the lowest
    highest_number = numbers[-1] #pull the last number which should be the largest
    return lowest_number, highest_number



def count_words(text: str) -> dict:
    """
    Count the frequency of each word in a text string.
    
    Args:
        text: Text string to analyze
    
    Returns:
        Dictionary with words as keys and counts as values
        
    Example:
        count_words("hello world hello")
        {'hello': 2, 'world': 1}
        count_words("")
        {}
    """
    # Take the argument, which is being stored as the variable text, and split it on the whitespace,
    # then store it in list_of_words
    list_of_words = text.split(" ")
    results = {} #creating an empty dictionary called results
    #Create a loop to go through our list of words
    for word in list_of_words:
        # If the word shows up in the dictionary, then increment the counter by one
        if word in results:
            results[word] += 1
        # If the word hasn't been seen before, add it to dictionary and set count to 1
        else:
            results[word] = 1
    return results



def is_palindrome(text: str) -> bool:
    """
    Check if a string is a palindrome (reads the same forwards and backwards).
    
    Args:
        text: String to check
    
    Returns:
        True if palindrome, False otherwise
        
    Example:
        is_palindrome("racecar")
        True
        is_palindrome("hello")
        False
        is_palindrome("A man a plan a canal Panama")
        True
    """
    pass


if __name__ == "__main__":
    name = "Mike"
    display_name(name)


    #define length and width variables
    length = 4
    width = 5
    #running the function returns area, which stores as my_area
    my_area = calculate_area(length, width)

    # Find Max Min Example
    #define a list of my numbers
    my_numbers = [2, 34, 16, 3, 4.5, 10.3, 21]
    #defining two new variables - lowest and highest.  Running the function returns lowest_number first,
    #which stores in to lowest.  It returns highest_number second, which stores into highest
    lowest, highest = find_max_min(my_numbers)


    # Word Count Example
    # define user words
    count_words("hello world hello")



    print("=== FUNCTION FUNDAMENTALS TESTS ===\n")

    # --- Test calculate_area ---
    assert calculate_area(5, 3) == 15, "Area calculation failed"


    # --- Test find_max_min ---
    assert find_max_min([1, 5, 3, 9, 2]) == (9, 1), "Max/min incorrect"
    assert find_max_min([-10, 0, 100]) == (100, -10), "Max/min incorrect"
    try:
        find_max_min([])
        raise AssertionError("Empty list should raise ValueError")
    except ValueError:
        pass

    # --- Test count_words ---
    assert count_words("hello world hello") == {'hello': 2, 'world': 1}, "Word count failed"
    assert count_words("") == {}, "Empty string should return empty dict"
    assert count_words("Hi hi HI") == {'hi': 3}, "Case normalization failed"

    # --- Test is_palindrome ---
    assert is_palindrome("racecar") is True, "Palindrome check failed"
    assert is_palindrome("hello") is False, "Palindrome check failed"
    assert is_palindrome("A man a plan a canal Panama") is True, "Palindrome phrase check failed"

    print("All tests passed successfully!")
