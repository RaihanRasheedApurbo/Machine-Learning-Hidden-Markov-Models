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

logger = logging.getLogger("task_1")
logger.setLevel(logging.ERROR)
logger.addHandler(stream_handler)


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
    logger.info(len(t1))
    logger.info(len(t2))
    logger.info(len(t1[0]))
    logger.info((t1[0][0]))
    logger.info((t1[1][0]))

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
    logger.info(len(observations))

    parameter_file = open(parameter_file_path)
    number_of_states = int(parameter_file.readline())
    logger.info(number_of_states)
    transition_matrix = []
    for i in range(number_of_states):
        transition_matrix.append([])
        line = parameter_file.readline()
        numbers = line.split()
        for number in numbers:
            transition_matrix[i].append(float(number))
        # logger.info(transition_matrix[i])
    logger.info(transition_matrix)

    means = []
    means_str = parameter_file.readline().split()
    for i in range(number_of_states):
        means.append(float(means_str[i]))
    logger.info(means)

    variances = []
    variances_str = parameter_file.readline().split()
    for i in range(number_of_states):
        variances.append(float(variances_str[i]))
    logger.info(variances)

    # initial probability p generation
    p = np.array(transition_matrix)
    a = np.array(transition_matrix)
    # supporting routine that matches two matrix values
    def match_matrix(a, b):
        if a.shape != b.shape:
            return False
        result = np.equal(a, b)
        # logger.info(result)
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
        # logger.info(np.matmul(a,p))
        # logger.info(p)
        if match_matrix(p, np.matmul(a, p)):
            break

    logger.info(p)
    logger.info(np.matmul(a, p))
    logger.info(iteration_needed_to_converge)
    initial_probabilites = p[0]  # taking first row

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
            logger.info(return_value)
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
    logger.info(len(x))

    output_file = open("output1.txt", "w")
    for item in x:
        output_file.write(f'"{item}"\n')
    output_file.close()


def task_2():

    data_file_path = "Sample input and output for HMM/Input/data.txt"
    parameter_file_path = "Sample input and output for HMM/Input/parameters.txt.txt"

    data_file = open(data_file_path, "r")
    lines = data_file.readlines()
    observations = []
    for line in lines:
        observations.append(float(line))
    logger.info(len(observations))

    parameter_file = open(parameter_file_path)
    number_of_states = int(parameter_file.readline())
    logger.info(number_of_states)
    transition_matrix = []
    for i in range(number_of_states):
        transition_matrix.append([])
        line = parameter_file.readline()
        numbers = line.split()
        for number in numbers:
            transition_matrix[i].append(float(number))
        # logger.info(transition_matrix[i])
    logger.info(transition_matrix)
    state_names = ["El Nino", "La Nina"]

    means = []
    means_str = parameter_file.readline().split()
    for i in range(number_of_states):
        means.append(float(means_str[i]))
    logger.info(means)

    variances = []
    variances_str = parameter_file.readline().split()
    for i in range(number_of_states):
        variances.append(float(variances_str[i]))
    logger.info(variances)

    def match_matrix(a1, b1):
            a = np.array(a1)
            b = np.array(b1)
            if a.shape != b.shape:
                return False
            result = np.equal(a, b)
            # logger.info(result)
            # logger.info(result.ndim)
            dimension = result.ndim
            if dimension == 2:
                number_of_rows = len(result)
                for i in range(number_of_rows):
                    for value in result[i]:
                        if not value:
                            return False
            elif dimension == 1:
                for value in result:
                    if not value:
                        return False

            return True

    def get_initial_probabilities(transition_matrix):
        # initial probability p generation
        p = np.array(transition_matrix)
        a = np.array(transition_matrix)
        # supporting routine that matches two matrix values
        

        iteration_needed_to_converge = 0
        while True:
            iteration_needed_to_converge += 1
            p = np.matmul(a, p)
            # logger.info(np.matmul(a,p))
            # logger.info(p)
            if match_matrix(p, np.matmul(a, p)):
                break
        logger.info(p)
        logger.info(np.matmul(a, p))
        logger.info(iteration_needed_to_converge)
        return p[0]

    

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
            logger.info(return_value)
            return return_value
        else:
            raise "undefined emission"


    def forward(
        observations, initial_probabilites, emission_function, transition_probabilites
    ):
        k = len(state_names)
        t = len(observations)
        p = initial_probabilites
        b = emission_function
        y = observations
        a = transition_probabilites

        t1 = [[]] * k  # number of states amount of one d array

        normalization_factor = 0
        for i in range(k):
            t1[i] = [0] * t
            t1[i][0] = p[i] * b(i, y[0])  # using 0 indexing
            normalization_factor += t1[i][0]

        for i in range(k):
            t1[i][0] /= normalization_factor

        logger.info(len(t1))
        logger.info(len(t1[0]))
        logger.info((t1[0][0]))
        logger.info((t1[1][0]))

        for j in range(1, t):
            normalization_factor = 0
            for i in range(k):
                t1[i][j] = b(i, y[j])
                summation = 0
                for l in range(k):
                    summation += t1[l][j - 1] * a[l][i]
                t1[i][j] *= summation
                normalization_factor += t1[i][j]

            for i in range(k):
                t1[i][j] /= normalization_factor

        return t1

    def backward(observations, emission_function, transition_probabilites):
        k = len(state_names)
        t = len(observations)

        b = emission_function
        y = observations
        a = transition_probabilites

        t1 = [[]] * k  # number of states amount of one d array
        normalization_factor = 0
        for i in range(k):
            t1[i] = [0] * t
            t1[i][t - 1] = 1
            normalization_factor += t1[i][t - 1]

        for i in range(k):
            t1[i][t - 1] /= normalization_factor

        logger.info(len(t1))
        logger.info(len(t1[0]))
        logger.info((t1[0][0]))
        logger.info((t1[1][0]))

        for j in range(t - 1, 0, -1):
            normalization_factor = 0
            for i in range(k):
                t1[i][j - 1] = 0
                for l in range(k):
                    t1[i][j - 1] += t1[l][j] * a[i][l] * b(l, y[j])
                normalization_factor += t1[i][j - 1]

            for i in range(k):
                t1[i][j - 1] /= normalization_factor
        return t1

    def likelihood_of_states(a, b):
        k = len(a)
        t = len(a[0])

        y = [[]] * k
        for i in range(k):
            y[i] = [0] * t

        for j in range(t):
            denominator = 0
            for i in range(k):
                y[i][j] = a[i][j] * b[i][j]
                denominator += y[i][j]
            for i in range(k):
                y[i][j] /= denominator

        return y

    def likelihood_of_transition(a,b,transition_matrix,emission_function,observations):
        k = len(a)
        t = len(a[0])
        y = observations
        e = [[]] * k
        for i in range(k):
            e[i] = [[]] * k
            for j in range(k):
                e[i][j] = [0] * (t-1)

        for l in range(t-1):
            denominator = 0
            for i in range(k):
                for j in range(k):
                    e[i][j][l] = a[i][l] * transition_matrix[i][j] * b[j][l+1] * emission_function(j,y[l+1])
                    denominator += e[i][j][l]
            for i in range(k):
                for j in range(k):
                    e[i][j][l] /= denominator 
        
        return e
                    
    def get_new_transition_matrix(e):
        k = len(e)
        t = len(e[0][0])
        t1 = [[]] * k
        for i in range(k):
            t1[i] = [0] * k
        
        for i in range(k):
            normalization_factor = 0
            for j in range(k):
                t1[i][j] = 0
                for l in range(t):
                    t1[i][j] += e[i][j][l]
                normalization_factor += t1[i][j]
            for j in range(k):
                t1[i][j] /= normalization_factor
        


        return t1

    def get_new_means(y,observations):
        k = len(y)
        t = len(y[0])

        means = [0] * k
        
        for i in range(k):
            sum1 = 0
            sum2 =0
            for j in range(t):
                sum1 += y[i][j] * observations[j]
                sum2 += y[i][j]
            means[i] = sum1/sum2
        return means

    def get_new_varience(y,observations,new_means):
        k = len(y)
        t = len(y[0])

        variences = [0] * k
        
        for i in range(k):
            sum1 = 0
            sum2 =0
            for j in range(t):
                sum1 += y[i][j] * (observations[j]-new_means[i])**2
                sum2 += y[i][j]
            variences[i] = sum1/sum2
        return variences

    counter = 0
    while True:

        initial_probabilites = get_initial_probabilities(transition_matrix)

        a = forward(
            observations, initial_probabilites, emission_function, transition_matrix
        )

        b = backward(observations, emission_function, transition_matrix)

        y = likelihood_of_states(a, b)

        e = likelihood_of_transition(a,b,transition_matrix,emission_function, observations)

        new_transition_matrx = get_new_transition_matrix(e)
        logger.info(new_transition_matrx)
        

        new_means = get_new_means(y,observations)
        logger.info(new_means)
        

        new_variences = get_new_varience(y,observations,new_means)
        logger.info(new_variences)
        
        counter += 1
        # logger.info(e[0][0][0])
        # logger.info(e[0][1][0])
        # logger.info(e[1][0][0])
        # logger.info(e[1][1][0])
        result1 = match_matrix(transition_matrix,new_transition_matrx)

        result2 = match_matrix(means,new_means)

        result3 = match_matrix(variances,new_variences)
        
        if result1 and result2 and result3:
            break
        
        transition_matrix = new_transition_matrx
        means = new_means
        variances = new_variences

    print("iteration taken to converge:", counter)
    print("transition probabilities:", transition_matrix)
    print("means:", means)
    print("variances:", variances)
    print("stationary probabilities:", initial_probabilites)

    x = viterbi(
        observations,
        initial_probabilites,
        emission_function,
        transition_matrix,
        state_names,
    )
    logger.info(len(x))

    output_file = open("output2.txt", "w")
    for item in x:
        output_file.write(f'"{item}"\n')
    output_file.close()

    output_file = open("learned_parameters.txt", "w")
    output_file.write(f"{len(transition_matrix)}\n")
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix[i])):
            output_file.write(f"{transition_matrix[i][j]} ")
        output_file.write(f"\n")
    for i in range(len(means)):
        output_file.write(f"{means[i]} ")
    output_file.write(f"\n")
    for i in range(len(variances)):
        output_file.write(f"{variances[i]} ")
    output_file.write(f"\n")
    for i in range(len(initial_probabilites)):
        output_file.write(f"{initial_probabilites[i]} ")
    output_file.write(f"\n")
    output_file.close()
    
    from hmmlearn import hmm

    remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
    observation_cast_2D = [] 
    for i in range(len(observations)):
        t = []
        t.append(observations[i])
        observation_cast_2D.append(t)
    remodel.fit(observation_cast_2D)
    print("***using hmmlearn package***")
    print("transition probabilities:", remodel.transmat_)
    print("means:", remodel.means_)
    print("variances:", remodel.covars_)
    
        
        


if __name__ == "__main__":
    # task_1()
    task_2()
