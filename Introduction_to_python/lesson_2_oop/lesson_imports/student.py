# Create a new class called student that includes
# Student Name
# Student ID
# Student email
# Assignment grades - dictionary where keys are assignment names and value is another dictionary with assignment grade and type

class Student:
    # for the dictionary, the main one is a string key with a value of another dictionary, which has a string key and values of either float or string
    def __init__(self, name: str, id: str, email: str, assignment_grades: dict[str, dict[str, float | str]]) -> None:
        self.name = name
        self.id = id
        self.email = email
        self.assignment_grades = assignment_grades

    # Make methods for getting:
    # Overall grade returning a float
    # Is passing returning a boolean

    def overall_grade(self):
        """
        This method will calculate the overall grade of the student.

        Returns: student grade

        """
        #overall grade is the sum of all grades divided by the number of grades
        student_grade = self.assignment_grades



# Create an instance of a student
if __name__ == '__main__'
    assignment_grades = {
        "Assignment 1": {
            "grade": 90
            "type": 'hw'
        }
        "Assignment 2": {
            "grade": 70
            "type": 'hw'
        }
        "Assignment 3": {
            "grade": 80
            "type": 'hw'
        }
        "Assignment 4": {
            "grade": 90
            "type": 'test'
        }
    }

# Create an instance of the student class with values for the arguments
mike = Student(name = "Mike", id = "123456", email = "<msorola@school.edu>", assignment_grades = assignment_grades)
print()
