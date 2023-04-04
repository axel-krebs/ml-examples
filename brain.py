# library
from abc import ABC, abstractmethod
from numbers import Number
import numpy as np
from typing import Any, Callable


class ActivationFunc(ABC):
    """The superclass (or 'interface') of all functions used for ML;
       Its one and only method is eval(): take Any input (vector, array, tensor)
       and produce Any output (single object or list type).

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def eval(self, input: Any) -> Any:
        raise "Abstract - Must implement!"

# Classification algorithms


class Classification(ActivationFunc):
    """Classification 'classifies' the input data according to their 'fitness' to the given classes.
       A class can be anything from real-world object to nearest neighbor (numbers), depending on
       what the input data denotes.

    Args:
        ActivationFunc (_type_): _description_
    """
    @abstractmethod
    def eval(self, input: Any) -> Any: # Any bcs. it could be a recognized image, voice recording etc.
        raise "Abstract - Must implement!"


class BinaryClassification(Classification):
    """Type of classification that returns only either 1 (True) or 0 (False)

    Args:
        Classification (_type_): _description_
    """

    @abstractmethod
    def eval(self, input: tuple[Any, ...]) -> float:  # 0 or 1 = True or False
        raise "Abstract - Write your own!"


class MultiGroupClassFunc(Classification):
    """Multi group classifiers 'know' about groups of classes, e.g. cats and dogs. 
       The classes match a certain pattern, which can bel learned.
       The eval() method applies the patterns to the input, which must be vectorized. 

    Args:
        Classification (_type_): _description_
    """

    @abstractmethod
    def eval(self, input: np.ndarray) -> int:  # 0 or 1 = True or False
        raise "Abstract - Write your own!"


class MultiLabelClassFunc(Classification):

    pass

# examples for the 'static' implementations of binary classifiers (cannot learn!)


class AND(BinaryClassification):

    def eval(self, input: tuple[int, int]) -> int:
        if input[0] == 1 and input[1] == 1:  # omit more elegant tuple summation for readability
            return 1
        return 0


class OR(BinaryClassification):

    def eval(self, input: tuple[int, int]) -> int:
        if input[0] == 0 and input[1] == 0:  # omit more elegant tuple summation for readability
            return 0
        return 1


class XOR(BinaryClassification):

    # This will throw exception "Can't instantiate abstract class XOR with abstract method eval() etc."
    # def eval(self, value: tuple[int, int]) -> int:
    pass


# Regression algorithms


class Regression(ActivationFunc):

    def eval(self, input: Any) -> Any:
        raise "Abstract - Must implement!"


class LogisticRegression(Regression):  # "Sigmoid"

    def eval(self, input: tuple[float, ...]) -> float:
        pass


class RectifiedLinearUnit(Regression):  # ReLU

    def eval(self, input: tuple[float, ...]) -> float:
        pass


class HyperbolicTangentFunc(Regression):

    def eval(self, input: tuple[float, ...]) -> float:
        pass


# Examples for 'static' regression functions (cannot learn!)

class MEAN(Regression):

    def eval(self, input: tuple[float, ...]) -> float:
        return sum(input) / len(input)


class MEDIAN(Regression):

    def eval(self, input: tuple[float, ...]) -> float:
        inp_len = len(input)
        inp_sorted = sorted(input, key=float)
        if inp_len % 2 != 0:
            return inp_sorted[inp_len // 2]
        else:
            index_of_upper_median = inp_len // 2
            index_of_lower_median = index_of_upper_median - 1
            lower_number = inp_sorted[index_of_lower_median]
            upper_number = inp_sorted[index_of_upper_median]
            return (lower_number + upper_number) / 2

# Nodes and Trees


class Node(ABC):
    """Superclass of all 'neurons'; after all, through 'annotation erasure' of Python
       it could be the only (instantiable) Singleton..

    Args:
        ABC (_type_): Marked abstract
    """

    def __init__(self, activation_func: ActivationFunc) -> None:
        self.a_func = activation_func

    def process_input(self, input: list[Any]) -> Any:
        return self.a_func.eval(input)


class Perceptron(Node):
    """Special case of a neuron that takes one or more inputs and puts out a _single_ 
       value - therefore it takes a binary classification activation function.
    """

    # Ctor only for signature (BinaryClassification)!
    def __init__(self, classifying_func: BinaryClassification) -> None:
        super().__init__(classifying_func)

    # IMPORTANT! return value is 0 or 1 (signature problem, actually disregarded by Python)!
    def process_input(self, input: tuple[float, ...]) -> int:
        return super().process_input(input)


class Predictron(Node):
    """Predicts an output value based on the input values.

    Args:
        Node (_type_): _description_
    """

    # Ctor only for signature annotation (must be Regression algorithm)!
    def __init__(self, regression_func: Regression) -> None:
        super().__init__(regression_func)

    # important! return value is a single float (the predicted value)!!
    def process_input(self, input: tuple[float, ...]) -> float:
        return super().process_input(input)


class DecisionTree:
    pass


def test():
    print("Running static tests..")

    # Test a Perceptron with binary classifier AND
    print("Trying Perceptron with static AND..")
    p_1 = Perceptron(AND())
    assert p_1.process_input((0, 1)) == 0
    assert p_1.process_input((1, 0)) == 0
    assert p_1.process_input((0, 0)) == 0
    assert p_1.process_input((1, 1)) == 1
    print("Success!")

    print("Trying Perceptron with static OR..")
    p_2 = Perceptron(OR())
    assert p_2.process_input((0, 1)) == 1
    assert p_2.process_input((1, 0)) == 1
    assert p_2.process_input((0, 0)) == 0
    assert p_2.process_input((1, 1)) == 1
    print("Success!")
    
    print("Trying Predictron with static MEDIAN..")
    p_median = Predictron(MEDIAN())
    assert p_median.process_input((1, 2, 3, 4, 5)) == 3
    assert p_median.process_input((1, 2, 4, 5)) == 3
    assert p_median.process_input((1, 11, 12, 13, 12345)) == 12 # an 'outlier'
    assert p_median.process_input((10, 11, 13, 14)) == 12
    print("Success!")

if __name__ == "__main__":
    print("This module is ot meant to executed directly; will run tests instead..")
    test()