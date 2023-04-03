# calculate single neural cell
from enum import Enum

matrix = ((0, 0), (0, 1), (1, 0), (1, 1))


def _and(x, y) -> int:
    if x == 0 and y == 0:
        return 0
    elif x == 0 and y == 1:
        return 0
    elif x == 1 and y == 0:
        return 0
    else:  # x == 1 AND y == 1
        return 1


def _or(x, y) -> int:
    if x == 0 and y == 0:
        return 0
    elif x == 0 and y == 1:
        return 1
    elif x == 1 and y == 0:
        return 1
    else:  # x == 1 AND y == 1
        return 1


def _xor(x, y) -> int:
    if x == 0 and y == 0:
        return 1
    elif x == 0 and y == 1:
        return 0
    elif x == 1 and y == 0:
        return 0
    else:  # x == 1 AND y == 1
        return 1


class Gate(Enum):
    AND = [(x, y, _and(x, y)) for x, y in matrix],
    OR = [(x, y, _or(x, y)) for x, y in matrix],
    XOR = [(x, y, _xor(x, y)) for x, y in matrix]

    def generate_epoch_table(self):
        """Generate a stepping table from truth values

        Returns:
            list: A list of four dicts corresponding to the result values 
        """
        epoch = []  # A table that represents a 'truth table'
        for table in self.value:
            for step in table:
                val = {
                    "x_1": step[0],
                    "x_2": step[1],
                    "y_output": step[2],
                    "y_actual": None,  # values depend on parameters!
                    "e_loss": None,
                    "W_1": None,
                    "W_2": None}
                epoch.append(val)
        return epoch


def step(x_1, x_2, weight_x1, weight_x2,  bias):
    return ((x_1 * weight_x1 + x_2*weight_x2) - bias)


class Perceptron():
    """Calculate the weight based on a given Bias and learning rate ("alpha").
       An initial weight must be given for x and y; moreover, a maximal depth 
       (maximal count of epochs) ensures that the loop will not be endless; 
       1000 epochs is the default. 
    """

    def __init__(self,
                 bias: float,
                 learning_rate: float,
                 initial_weight_x1: float,
                 initial_weight_x2: float,
                 max_depth: int = 1000) -> None:
        self._bias = bias
        self._alpha = learning_rate
        self._weight_x1 = initial_weight_x1
        self._weight_x2 = initial_weight_x2
        self._max_depth = max_depth
        self._epochs = []  # already calculated epochs; order matters!

    def calc(self, truth_table: Gate):
        
        # ensure we're not getting an endless loop
        while len(self._epochs) <= self._max_depth:
            new_epoch = truth_table.generate_epoch_table()
            has_errors = False
            for entry in new_epoch:

                # 1. calculate step ("Activate function", y_actual)
                step_res = step(entry["x_1"], entry["x_2"], self._weight_x1,
                                self._weight_x2, self._bias)
                # only 0 or 1 are allowed (the step function serves debugging purposes mainly)
                entry["y_actual"] = round(step_res)

                # 2. calculate error ("Loss function")
                entry["e_loss"] = entry["y_output"] - entry["y_actual"]
                if entry["e_loss"] != 0:
                    has_errors = True

                # 3. Update weights (Optimizer)
                w1_delta = self._alpha * entry["x_1"] * entry["e_loss"]
                w2_delta = self._alpha * entry["x_2"] * entry["e_loss"]
                entry["W_1"] = self._weight_x1 + w1_delta
                entry["W_2"] = self._weight_x2 + w2_delta

            self._epochs.append(new_epoch)

            # Now, for the next iteration, we'll use the new weight values from the last iteration
            self._weight_x1 = entry["W_1"]
            self._weight_x2 = entry["W_2"]

            # if epoch contains no errors, we're finished (we could as well iterate the 'error' field)
            if has_errors == False:
                print("Hurray! We've found a new weight, after",
                      len(self._epochs), "epochs.")
                break
        else:
            print("Max-depth reached!")


def print_menu():
    print("0. Leave the program.")
    print("1. Calc AND gate.")
    print("2. Calc OR gate.")
    print("3. Calc XOR gate.")


def main():
    
    print("Perceptron")
    
    while True:
        
        print_menu()
        
        user_inp = input(">")
        if user_inp == "0":
            break
        
        elif user_inp == "1":
            p = Perceptron(0.2, 0.1, 0.3, -0.1, 100)
            p.calc(Gate.AND)
            
        elif user_inp == "2":
            p = Perceptron(0.2, 0.1, 0.3, -0.1)
            p.calc(Gate.OR)
            
        elif user_inp == "3":
            print(
                "A list  of available literature will be printed to the output manager..")


main()
