
# from the lesson imports folder --> rectangle file, import the class Rectangle
from lesson_imports.rectangle import Rectangle

# Since this is the file we will run the program from, it will be recognized as __main__
# Since this is __main__, this code will run
if __name__ == '__main__':

    # Create an instance of our class Rectangle
    blue_rectangle = Rectangle(length = 10, width = 10, color = 'blue')
    print()