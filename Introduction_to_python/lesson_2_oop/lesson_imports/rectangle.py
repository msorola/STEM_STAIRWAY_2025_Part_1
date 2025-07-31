

# give the class a name
class Rectangle:

    # Some attributes of a rectangle are length, width, and color
    # We want to define the attributes as arguments
    def __init__(self, width: float, length: float, color: str):

        # Set the attributes of the class
        self.width = width
        self.length = length
        self.color = color

    # make a method to find area of the rectangle, with self as the argument since we made it above
    def get_area(self):
        """
        This method returns the area of the rectangle.

        Arguments:
            None

        Returns:
            Area (float): The area of the rectangle.

        """
        area = self.width * self.length
        return area

    # Make a method for calculating perimeter
    def get_perimeter(self):
        perimeter = self.width * 2 + self.length * 2
        return perimeter

    # Make a method to check to see if it is a square that will return T or F
    def is_square(self):
        # If length and width are equal, it must be a square
        if self.width == self.length:
            return True
        else:
            return False


