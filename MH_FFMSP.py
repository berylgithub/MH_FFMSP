import numpy as np
import time
from docplex.mp.model import Model # CPLEX lib

def ffmsp_obj(sol, data, threshold):
    '''objective function of the FFMSP:
    computes the hamming distance of the solution and each solution in the data,
    if the count of hamming distances of sol and one sol from data >= t, then add 1 cardinality.
    
    returns: objective function value, scalar.
    params:
        - sol, vector of solution of FFMSP, shape = m
        - data, matrix containing list of strings, shape = (n, m)
        - threshold, [0, m], scalar.
    '''
    # init vars:
    #n = data.shape[0]; m = data.shape[1]
    
    # compute the hamming distance of one cell of sol to all in data:
    hamming_array = np.not_equal(sol, data) # contains matrix of predicate function
    # much faster, by the factor of 10:
    count = np.sum(hamming_array, axis=1) # sum the differences
    count = np.where(count >= threshold) # count the differences where difference > threshold
    y = count[0].shape[0] # get the hamming count
    '''
    # slow:
    for i in range(n):
        count_hamming = 1 if np.sum(hamming_array[i]) >= threshold else 0
        y += count_hamming
    '''
    return y

def greedy(data, alphabet, t):
    '''
    the simple greedy algo for FFMSP uses a consensus of each column (string position): 
    takes the char with the least occurence from all strings on each position - maximization problem. 
    
    returns:
        - solution vector, shape = m
        - objective function, scalar
    params:
        - data, matrix containing list of strings, shape = (n, m)
        - alphabet, vector (mathematically, a set), shape = len(alphabet)
        - threshold t, scalar
    '''
    n = data.shape[0]; m = data.shape[1]; alpha_size = len(alphabet)
    threshold = int(t*m) # since t is in percentage
    freq_mat = np.zeros((alpha_size, m))
    # count the occurences of each alphabet column wise:
    for i in range(alpha_size):
        freq_mat[i] = np.count_nonzero(data == alphabet[i], axis = 0) # alphabet[i] == i in this case, can use whichever
    
    #print(freq_mat)
    sol = np.argmin(freq_mat, axis=0) # get char with lowest frequency for each position [0, m]
    f = ffmsp_obj(sol, data, threshold) # compute obj fun
    return sol, f

def local_search(data, alphabet, t, init='greedy', sol=None):
    '''
    simple local search, by flipping each cell and only accepting solution(s) that are better
    pretty much gradient descent but FFMSP ver.
    
    returns the same params and takes in the same params as greedy algo except:
    params:
        - init, decides initialization mode, string
        - sol, starting solution, will use this instead of init if this is not empty, shape = m
    '''
    # init var:
    n = data.shape[0]; m = data.shape[1]; alpha_size = len(alphabet)
    threshold = int(t*m) # for obj fun
    f=0
    
    # generate init sol:
    if sol is not None:
        f = ffmsp_obj(sol, data, threshold)
    else:
        if init == "greedy":
            sol, f = greedy(data, alphabet, t)
        elif init == "random":
            sol = np.random.randint(alpha_size, size=m)
            f = ffmsp_obj(sol, data, threshold)
    
    # do local search, flip each bit position:
    for i in range(m):
        for j in range(alpha_size):
            if sol[i] != alphabet[j]: # exclude current char
                # implicitly flip bit, check if better - if yes then stop and check next pos:
                # slow boi:
                #sol_new = np.copy(sol) # need to copy since numpy by default refers to memory instead. Need to replace with more eficient op
                #sol_new[i] = j
                #f_new = ffmsp_obj(sol_new, data, threshold)
                # more efficient one:
                pre_flip = sol[i] # store the unflipped bit
                sol[i] = j # flip the bit
                f_new = ffmsp_obj(sol, data, threshold)
                #print(sol_new, f_new)
                #print(sol, f, i, j, sol_new, f_new)
                if f_new >= f:
                    #sol = sol_new; f = f_new
                    f = f_new
                    #break # without break seems better
                else:
                    sol[i] = pre_flip # put back the old bit
    return sol, f

def metaheuristic(data, alphabet, t, max_loop, rand, init="greedy"):
    '''
    simple idea: to avoid local minima, use perturbation - randomly swapping cells with random letters, where the random percentage is a tuning parameter  
    returns the same params as the greedy algo.
    params, same as prev methods, except:
        - max_loop, hyperparameter determining maximum perturbation+local search ops, int scalar
        - rand, a hyperparameter [0,1] percentage of the mutated cells, lower means faster convergence, real scalar 
            -> seems like near 0 is a good choice
    '''
     # init var:
    n = data.shape[0]; m = data.shape[1]; alpha_size = len(alphabet)
    threshold = int(t*m) # for obj fun
    f=0
    
    # generate init sol:
    if init == "greedy":
        sol, f = greedy(data, alphabet, t)
    elif init == "random":
        sol = np.random.randint(alpha_size, size=m)
        f = ffmsp_obj(sol, data, threshold)
        
    # loop the local search and perturbation:
    i = 0
    perturb_length = int(rand*m)
    print(perturb_length)
    # do initial local search for the lower bound:
    sol, f = local_search(data, alphabet, t, sol=sol)
    while i<max_loop:
        # perturb sol, generate random integer array [0,alpha_size] with length = perturb_length which replaces random cells:
        part_sol = np.random.randint(alpha_size, size=perturb_length)
        idx = np.random.randint(m, size=perturb_length) # replacement indexes
        #print(part_sol, idx)
        
        # slow ver:
        sol_perturb = np.copy(sol)
        sol_perturb[idx] = part_sol # replace some sol components wiwth the part_sol
        #print(sol_perturb)
        # do local search:
        sol_new, f_new = local_search(data, alphabet, t, sol=sol_perturb)
        '''
        # more efficient:
        sol[idx] = part_sol
        sol, f_new = local_search(data, alphabet, t, sol=sol)
        '''
        #sort of greedy acceptante criteria, compare with previous local minimum:
        if f_new >= f:
            sol = sol_new
            f = f_new
            print(i,"accepted",f)
        i+=1
        #yield sol, f
    return sol, f

'''
=== directly timed version funs ===
'''
def metaheuristic_wt(data, alphabet, t, max_loop, rand, init="greedy", time_limit=90):
    '''
    timed version
    params, same as prev methods, except:
        - max_loop, hyperparameter determining maximum perturbation+local search ops, int scalar
        - rand, a hyperparameter [0,1] percentage of the mutated cells, lower means faster convergence, real scalar 
            -> seems like near 0 is a good choice
    '''
    # init var:
    m = data.shape[1]; alpha_size = len(alphabet)
    threshold = int(t*m) # for obj fun
    f=0
    
    # generate init sol:
    if init == "greedy":
        sol, f = greedy(data, alphabet, t)
    elif init == "random":
        sol = np.random.randint(alpha_size, size=m)
        f = ffmsp_obj(sol, data, threshold)
        
    # start timer:
    start = time.time()
    
    # loop the local search and perturbation:
    i = 0
    perturb_length = int(rand*m)
    #print(perturb_length)
    # do initial local search for the lower bound:
    sol, f = local_search(data, alphabet, t, sol=sol)
    while i<max_loop:
        # check fi it's already time limit:
        if time.time() - start >= time_limit: # > since there will be always time artefact (in ms)
            #print(time.time() - start,"s")
            #print("last sol", sol, f)
            break
        # perturb sol, generate random integer array [0,alpha_size] with length = perturb_length which replaces random cells:
        part_sol = np.random.randint(alpha_size, size=perturb_length)
        idx = np.random.randint(m, size=perturb_length) # replacement indexes
        #print(part_sol, idx)
        
        # slow ver:
        sol_perturb = np.copy(sol)
        sol_perturb[idx] = part_sol # replace some sol components wiwth the part_sol
        #print(sol_perturb)
        # do local search:
        sol_new, f_new = local_search(data, alphabet, t, sol=sol_perturb)
        '''
        # more efficient:
        sol[idx] = part_sol
        sol, f_new = local_search(data, alphabet, t, sol=sol)
        '''
        #sort of greedy acceptante criteria, compare with previous local minimum:
        if f_new >= f:
            sol = sol_new
            f = f_new
            #print(i,"accepted",f)
        i+=1
        #yield sol, f
    return sol, f

def cplex_ffmsp(data, alphabet, t, time_limit=90):
    '''
    CPLEX version opt for FFMSP, in ILP formulation.
    the parameters are the subset parameters of MH algo ver.
    '''
    n = data.shape[0]; m = data.shape[1]; alpha_size = len(alphabet)
    threshold = int(t*m)

    model = Model(name='FFMSP')
    y = model.binary_var_list(n, name="y") # objective var, length n vector
    x = model.binary_var_matrix(m, alpha_size, name="x") # matrix with (m, alpha_size) shape
    s = data.T # according to the ILP formulation in the .pdf, for clarity, matrix with (m, n) shape

    # m equality constraints:
    for i in range(m):
        model.add_constraint(model.sum(x[i,j] for j in range(alpha_size)) == 1)
    
    # n inequality constraints:
    for r in range(n):
        model.add_constraint(model.sum(x[i,s[i,r]] for i in range(m)) <= m - threshold*y[r])
    #print(threshold*y[0])
    # objective function:
    obj = model.sum(y)

    # set everything:
    model.set_objective('max', obj)
    model.set_time_limit(time_limit)
    model.print_information()    
    print(model.export_as_lp_string())

    # solve:
    model.solve()
    print("obj = ",model.objective_value)
    model.print_solution()

    #
if __name__ == "__main__":
    def unit_test():
        '''individual tests'''
        filename = "problem_instances/100-300-001.txt"
        data = []
        mapper = {'A':0, 'C':1, 'T':2, 'G':3} #char to int
        rev_mapper = {0:'A', 1:'C' , 2:'T', 3:'G'} #int to char, whenever needed
        alphabet = (0,1,2,3)
        # read per char, for matrix data structure, while mapping ['A', 'C', 'T', 'G'] to [0,1,2,3] at the same time:
        with open(filename) as fileobj:
            for line in fileobj:
                d = []
                line = line.rstrip("\n")
                for ch in line:
                    mapch = mapper[ch]
                    d.append(mapch)
                data.append(d)
        n = len(data); m = len(data[0])
        data = np.array(data)
        #count = np.char.count(data[0], 'A')
        count = np.count_nonzero(data == mapper['A'], axis=0)

        #ffmsp_obj(np.random.randint(0, 4, 300), data, 0.8*300)
        '''
        sol = np.array([0,1,3,3])
        dat = np.array([
                        [0, 1, 3, 3],
                        [0, 2, 0, 2],
                        [3, 1, 3, 0],
                        [0,2,3,3]
                        ])
        print(ffmsp_obj(sol, dat, 4))
        '''
        '''
        start = time.time()
        #metaheuristic(data, alphabet, 0.8, 100, 1e-2, init="greedy")
        sol, f = metaheuristic_wt(data, alphabet, 0.8, int(1e7), 1e-2, init="greedy", time_limit=90)
        print(sol, f)
        elapsed = time.time()-start
        print(elapsed)
        '''
        cplex_ffmsp(data, alphabet, 0.8, time_limit=5)
    
    def evaluation_statistics():
        ''' do the algorithms evaluation across all test instances then compute the statistics'''
        print()

    '''=== actual main starts here ==='''
    unit_test()