from numpy import exp, array, random, dot

class NeuralNetwork():
    # This is the neural network initializer function
    def __init__(self):
        # use the .seed function to make sure to use the same random number
        # Every time the program runs
        random.seed(1)

        # model a single network with four input nodes and one ouput nodes
        # assign random weights to a 4 x 1 matrix with values between -1 and 1 and with mean 0
        self.synaptic_weights = 2 * random.random((4 , 1)) - 1

    # This is the sigmoid function, which when graphed are S shaped curves
    # This is the function we use to normalise the data
    # There are many sigmoid functions such as arctan(x), tanh(x) and smoothstep function
    # For this case we are using a logistic function to normalise the data so that we get an output between
    # 0 and 1
    # The function is 1/1+e^(-x)

    def __sigmoid(self, x):
        return 1 / (1+exp(-x))

    # This next function is to find the derivative of the sigmoid
    # The derivative will be the gradient of the curve, aka gradient descent
    # This will show how confident we are about the weight on the nodes

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # This is the fucntion to train our neural net, which we will be doing through trial and error
    def train(self, training_set_inputs, training_set_ouputs, training_set_iterations):
        for iteration in xrange(training_set_iterations):
            # Pass training data through the network
            output = self.think(training_set_inputs)

            # Calculate the error by subtracting desired ouputs from predicted outputs
            error = training_set_ouputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve
            # This means less confident weights are adjusted more
            # This means inputs, which are zero, do not cause changes to the weights
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights
            self.synaptic_weights += adjustment

    # The neural network calulates
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    # Initialize a neural network
    neural_net = NeuralNetwork()

    print "Random starting synaptic_weights: "
    print neural_net.synaptic_weights

    # This is the training set with 5 examples each consisting of 4 input values and 1 ouput value
    # First value is # of balls
    # Second value is # of gears
    # Third value is # of rotors: max = 4
    # Fourth value is feasibility rating from 1 to 3
    data = array([[36, 12, 4, 3], [0, 15, 4, 3], [5, 3, 1, 1], [300, 5, 2, 2], [200, 5, 4, 2]])
    ranking = array([[1, 2, 5, 4, 3]]).T

    # Train the neural net 10000 times
    neural_net.train(data, ranking, 10000)

    print "New weights after training: "
    print neural_net.synaptic_weights

    print "Considering new situation [0,0,0,1] -> ?: "
    print neural_net.think(array([40, 11, 4, 2]))
