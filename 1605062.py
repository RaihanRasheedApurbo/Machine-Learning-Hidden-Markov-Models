import logging
import random
import math
import numpy as np

# logger initialization
formatter = logging.Formatter(
    "\n*********Line no:%(lineno)d*********\n%(message)s\n***************************"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger_1 = logging.getLogger("task_1")
logger_1.setLevel(logging.INFO)
logger_1.addHandler(stream_handler)


def viterbi(
    observations,
    initial_probabilites,
    emission_function,
    transition_probabilites,
    state_names,
):
    k = len(state_names)
    t = len(observations)
    p = initial_probabilites
    b = emission_function
    y = observations
    a = transition_probabilites
    s = state_names

    t1 = [[]] * k  # number of states amount of one d array
    t2 = [[]] * k

    for i in range(k):
        t1[i] = [0] * t
        t1[i][0] = math.log(p[i]) + math.log(b(i, y[0]))  # using 0 indexing
        t2[i] = [0] * t
    logger_1.info(len(t1))
    logger_1.info(len(t2))
    logger_1.info(len(t1[0]))
    logger_1.info((t1[0][0]))
    logger_1.info((t1[1][0]))

    for j in range(1, t):
        for i in range(k):
            max_value = float("-inf")
            max_arg = -1
            for l in range(k):

                value = t1[l][j - 1] + math.log(a[l][i]) + math.log(b(i, y[j]))
                if value > max_value:
                    max_value = value
                    max_arg = l
            t1[i][j] = max_value
            t2[i][j] = max_arg

    max_arg = -1
    max_value = float("-inf")
    for i in range(k):
        value = t1[l][t - 1]
        if value > max_value:
            max_value = value
            max_arg = l

    z = [-1] * t
    z[t - 1] = max_arg  # 0 indexing hence Tth hidden state is inside t-1 index of x
    x = [-1] * t
    x[t - 1] = s[z[t - 1]]
    for j in range(t - 1, 0, -1):
        z[j - 1] = t2[z[j]][j]
        x[j - 1] = s[z[j - 1]]

    return x


def task_1():

    data_file_path = "Sample input and output for HMM/Input/data.txt"
    parameter_file_path = "Sample input and output for HMM/Input/parameters.txt.txt"

    data_file = open(data_file_path, "r")
    lines = data_file.readlines()
    observations = []
    for line in lines:
        observations.append(float(line))
    logger_1.info(len(observations))

    parameter_file = open(parameter_file_path)
    number_of_states = int(parameter_file.readline())
    logger_1.info(number_of_states)
    transition_matrix = []
    for i in range(number_of_states):
        transition_matrix.append([])
        line = parameter_file.readline()
        numbers = line.split()
        for number in numbers:
            transition_matrix[i].append(float(number))
        # logger_1.info(transition_matrix[i])
    logger_1.info(transition_matrix)

    means = []
    means_str = parameter_file.readline().split()
    for i in range(number_of_states):
        means.append(float(means_str[i]))
    logger_1.info(means)

    variances = []
    variances_str = parameter_file.readline().split()
    for i in range(number_of_states):
        variances.append(float(variances_str[i]))
    logger_1.info(variances)

    # initial probability p generation
    p = np.array(transition_matrix)
    a = np.array(transition_matrix)
    # supporting routine that matches two matrix values
    def match_matrix(a, b):
        if a.shape != b.shape:
            return False
        result = np.equal(a, b)
        # logger_1.info(result)
        number_of_rows = len(result)
        for i in range(number_of_rows):
            for value in result[i]:
                if not value:
                    return False
        return True

    iteration_needed_to_converge = 0
    while True:
        iteration_needed_to_converge += 1
        p = np.matmul(a, p)
        # logger_1.info(np.matmul(a,p))
        # logger_1.info(p)
        if match_matrix(p, np.matmul(a, p)):
            break

    logger_1.info(p)
    logger_1.info(np.matmul(a, p))
    logger_1.info(iteration_needed_to_converge)
    initial_probabilites = p[0]  # taking first row

    # input parameter o stands for observation space (possible emissions)
    # input parameter
    # def viterbi(s,p,y,a,m,v):
    #     t1 = [] # stores probabilites upto certain time
    #     t2 = [] # stores parent info
    #     for i in range(number_of_states):
    #         t1.append([])
    #         t1[i].append()

    def pdf_value(mean, variance, value):
        sigma_squared = variance
        sigma = math.sqrt(sigma_squared)
        pie = 3.1416
        e = 2.718
        exponent_of_e = -0.5 * ((value - mean) / sigma) ** 2
        pdf = pow(e, exponent_of_e) / (sigma * math.sqrt(2 * pie))
        return pdf

    def emission_function(i, value):
        if i == 0 or i == 1:
            return_value = pdf_value(means[i], variances[i], value)
            logger_1.info(return_value)
            return return_value
        else:
            raise "undefined emission"

    state_names = ["El Nino", "La Nina"]

    x = viterbi(
        observations,
        initial_probabilites,
        emission_function,
        transition_matrix,
        state_names,
    )
    logger_1.info(len(x))

    output_file = open("output1.txt", "w")
    for item in x:
        output_file.write(f'"{item}"\n')
    output_file.close()


if __name__ == "__main__":
    task_1()
