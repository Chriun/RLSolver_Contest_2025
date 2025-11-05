import tensorflow as tf
import numpy as np
import argparse
from math import sqrt
import os
import time
import random
from natsort import natsorted
from typing import Any, Optional, Union, Text, Sequence, Tuple, List

Tensor = Any

'''
Config
'''
class Config:
    def __init__(self, graph_path, seed):
        N, Jz = self.read_graph(graph_path)
        self.seed = seed

        self.N = N #total number of sites
        self.Jz = Jz

        '''model'''
        self.num_units = 40 #number of memory units
        self.num_layers = int(sqrt(N))
        self.activation_function = tf.nn.elu #non-linear activation function for the Tensorized RNN cell

        '''training'''
        self.numsamples = 50 #number of samples used for training
        self.lr = 5*(1e-4) #learning rate
        self.T0 = 2 #Initial temperature
        self.Bx0 = 0 #Initial magnetic field
        self.num_warmup_steps = 1000 #number of warmup steps
        self.num_annealing_steps = 2**8 #number of annealing steps
        self.num_equilibrium_steps = 5 #number of training steps after each annealing step
    
    def read_graph(self, graph_path):

        with open(graph_path, "r") as f:
            line = f.readline()
            is_first_line = True
            while line is not None and line != '':
                if is_first_line:
                    nodes, edges = line.strip().split(" ")
                    num_nodes = int(nodes)
                    num_edges = int(edges)
                    is_first_line = False
                    Jz = np.zeros((num_nodes, num_nodes), dtype=np.float64)
                else:
                    node1, node2, weight = line.strip().split(" ")
                    Jz[(int(node1)-1, int(node2)-1)] = float(weight)
                    Jz[(int(node2)-1, int(node1)-1)] = float(weight)
                line = f.readline()
        return num_nodes, Jz
    
'''
Network
'''
class DilatedRNNWavefunction(object):
    def __init__(self, systemsize, units=[2], layers=1, cell=tf.nn.rnn_cell.BasicRNNCell, activation=tf.nn.relu, scope='DilatedRNNwavefunction', seed=111):
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.N=systemsize #Number of sites of the 1D chain
        self.numlayers=layers
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                tf.set_random_seed(seed)  # tensorflow pseudo-random generator

                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                self.rnn=[[cell(num_units = units[i], activation = activation,name="rnn_"+str(n)+str(i),dtype=tf.float64) for n in range(self.N)] for i in range(self.numlayers)]
                self.dense = [tf.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense'+str(n)) for n in range(self.N)] #Define the Fully-Connected layer followed by a Softmax

    def sample(self,numsamples,inputdim):
        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            samples = []
            probs = []
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):

                inputs=tf.zeros((numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        # rnn_states.append(1.0-self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                        rnn_states.append(self.rnn[i][n].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i])

                    output=self.dense[n](rnn_output)
                    probs.append(output)
                    sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,]) #Sample from the probability
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim,dtype = tf.float64)

        probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1
        one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

        return self.samples,self.log_probs

    def log_probability(self,samples,inputdim):
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]

            inputs=tf.zeros((self.numsamples, self.inputdim), dtype=tf.float64)

            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                probs=[]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        rnn_states.append(self.rnn[i][n].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i])

                    output=self.dense[n](rnn_output)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs
  
'''
Utils
'''
def Fullyconnected_diagonal_matrixelements(Jz, samples):
    numsamples = samples.shape[0]
    N = samples.shape[1]
    energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1):
      values = np.expand_dims(samples[:,i], axis = -1)+samples[:,i+1:]
      valuesT = np.copy(values)
      valuesT[values==2] = +1 #If both spins are up
      valuesT[values==0] = +1 #If both spins are down
      valuesT[values==1] = -1 #If they are opposite

      energies += np.sum(valuesT*(-Jz[i,i+1:]), axis = 1)

    return energies

def Fullyconnected_localenergies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
    numsamples = samples.shape[0]
    N = samples.shape[1]
    
    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1):
      # for j in range(i+1,N):
      values = np.expand_dims(samples[:,i], axis = -1)+samples[:,i+1:]
      valuesT = np.copy(values)
      valuesT[values==2] = +1 #If both spins are up
      valuesT[values==0] = +1 #If both spins are down
      valuesT[values==1] = -1 #If they are opposite

      local_energies += np.sum(valuesT*(-Jz[i,i+1:]), axis = 1)

    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        count = 0
        for i in range(N-1):  #Non-diagonal elements
            valuesT = np.copy(samples)
            valuesT[:,i][samples[:,i]==1] = 0 #Flip spin i
            valuesT[:,i][samples[:,i]==0] = 1 #Flip spin i

            count += 1
            queue_samples[count] = valuesT

        len_sigmas = (N+1)*numsamples
        steps = len_sigmas//50000+1 #I want a maximum of 50000 in batch size just to be safe I don't allocate too much memory

        queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N])
        for i in range(steps):
          if i < steps-1:
              cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
          else:
              cut = slice((i*len_sigmas)//steps,len_sigmas)
          log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})


        log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
        for j in range(numsamples):
            local_energies[j] += -Bx*np.sum(0.5*(np.exp(log_probs_reshaped[1:,j]-log_probs_reshaped[0,j])))

    return local_energies

'''
Variational Classical Annealing
'''
def run_vca(config: Config):
    seed = config.seed
    tf.compat.v1.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

    N = config.N
    Jz = config.Jz

    num_units = config.num_units
    num_layers = config.num_layers
    activation_function = config.activation_function

    numsamples = config.numsamples
    lr = config.lr
    T0 = config.T0
    Bx0 = config.Bx0
    num_warmup_steps = config.num_warmup_steps
    num_annealing_steps = config.num_annealing_steps
    num_equilibrium_steps = config.num_equilibrium_steps

    print('\n')
    print("Number of spins =", N)
    print("Initial_temperature =", T0)
    print('Seed = ', seed)

    num_steps = num_annealing_steps*num_equilibrium_steps + num_warmup_steps

    print("\nNumber of annealing steps = {0}".format(num_annealing_steps))
    print("Number of training steps = {0}".format(num_steps))

    units = [num_units] * num_layers
    DRNNWF = DilatedRNNWavefunction(N, units=units, layers=num_layers, cell=tf.nn.rnn_cell.BasicRNNCell, activation=activation_function, seed=seed) #contains the graph with the RNNs
    with tf.compat.v1.variable_scope(DRNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with DRNNWF.graph.as_default():

            global_step = tf.Variable(0, trainable=False)
            learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
            learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)

            #Defining the optimizer
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate)

            #Defining Tensorflow placeholders
            Eloc=tf.compat.v1.placeholder(dtype=tf.float64,shape=[numsamples])
            sampleplaceholder_forgrad=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_probs_forgrad = DRNNWF.log_probability(sampleplaceholder_forgrad,inputdim=2)
            samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,N))
            log_probs_tensor=DRNNWF.log_probability(samples_placeholder,inputdim=2)
            samplesandprobs = DRNNWF.sample(numsamples=numsamples,inputdim=2)

            T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=())

            #Here we define a fake cost function that would allows to get the gradients of free energy using the tf.stop_gradient trick
            Floc = Eloc + T_placeholder*log_probs_forgrad
            cost = tf.reduce_mean(tf.multiply(log_probs_forgrad,tf.stop_gradient(Floc))) - tf.reduce_mean(log_probs_forgrad)*tf.reduce_mean(tf.stop_gradient(Floc))

            gradients, variables = zip(*optimizer.compute_gradients(cost))
            #Calculate Gradients---------------

            #Define the optimization step
            optstep=optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)

            #Tensorflow saver to checkpoint
            saver=tf.compat.v1.train.Saver()

            #For initialization
            init=tf.compat.v1.global_variables_initializer()
            initialize_parameters = tf.initialize_all_variables()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.compat.v1.Session(graph=DRNNWF.graph, config=config)
    sess.run(init)

    ## Run Variational Annealing
    with tf.compat.v1.variable_scope(DRNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with DRNNWF.graph.as_default():
            #To store data
            meanEnergy=[]
            varEnergy=[]
            varFreeEnergy = []
            meanFreeEnergy = []
            samples = np.ones((numsamples, N), dtype=np.int32)
            queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32)
            log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) 

            T = T0 #initializing temperature
            Bx = Bx0 #initializing magnetic field

            sess.run(initialize_parameters) #Reinitialize the parameters

            start = time.time()
            for it in range(len(meanEnergy),num_steps+1):
                #Annealing
                if it>=num_warmup_steps and  it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it % num_equilibrium_steps == 0:
                    annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
                    T = T0*(1-annealing_step/num_annealing_steps)
                    Bx = Bx0*(1-annealing_step/num_annealing_steps)

                #Showing current status after that the annealing starts
                if it%num_equilibrium_steps==0:
                    if it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it>=num_warmup_steps:
                        annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
                        print("\nAnnealing step: {0}/{1}".format(annealing_step,num_annealing_steps))

                samples, log_probabilities = sess.run(samplesandprobs)

                # Estimating the local energies
                local_energies = Fullyconnected_localenergies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess)

                meanE = np.mean(local_energies)
                varE = np.var(local_energies)

                #adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)

                meanF = np.mean(local_energies+T*log_probabilities)
                varF = np.var(local_energies+T*log_probabilities)

                meanFreeEnergy.append(meanF)
                varFreeEnergy.append(varF)

                if it%num_equilibrium_steps==0:
                    print('mean(E): {0}, mean(F): {1}, var(E): {2}, var(F): {3}, #samples {4}, #Training step {5}'.format(meanE,meanF,varE,varF,numsamples, it))
                    print("Temperature: ", T)
                    print("Magnetic field: ", Bx)

                #Here we produce samples at the end of annealing
                if it == num_annealing_steps*num_equilibrium_steps + num_warmup_steps:

                    Nsteps = 20
                    numsamples_estimation = 10**6 #Num samples to be obtained at the end
                    numsamples_perstep = numsamples_estimation//Nsteps #The number of steps taken to get "numsamples_estimation" samples (to avoid memory allocation issues)

                    samplesandprobs_final = DRNNWF.sample(numsamples=numsamples_perstep,inputdim=2)
                    energies = np.zeros((numsamples_estimation))
                    solutions = np.zeros((numsamples_estimation, N))
                    print("\nSaving energy and variance before the end of annealing")

                    for i in range(Nsteps):
                        # print("\nsampling started")
                        samples_final, _ = sess.run(samplesandprobs_final)
                        # print("\nsampling finished")
                        energies[i*numsamples_perstep:(i+1)*numsamples_perstep] = Fullyconnected_diagonal_matrixelements(Jz,samples_final)
                        solutions[i*numsamples_perstep:(i+1)*numsamples_perstep] = samples_final
                        print("Sampling step:" , i+1, "/", Nsteps)
                    print("meanE = ", np.mean(energies))
                    print("varE = ", np.var(energies))
                    print("minE = ",np.min(energies))
                    return np.mean(energies), np.min(energies)

                #Run gradient descent step
                sess.run(optstep,feed_dict={Eloc:local_energies, sampleplaceholder_forgrad: samples, learningrate_placeholder: lr, T_placeholder:T})

                if it%5 == 0:
                    print("Elapsed time is =", time.time()-start, " seconds")
                    print('\n\n')
            
            

'''
Run VCA
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_instance", type=str,
                        help="input the data file for the problem instance")

    args = parser.parse_args()
    data_dir = args.problem_instance
    input_files = [ f for f in os.listdir(data_dir) ]
    input_files = natsorted(input_files)

    results_dir = f"results/{'/'.join(data_dir.split('/')[1:])}"
    os.makedirs(results_dir, exist_ok=True)

    i = 0
    for file in input_files:
        i += 1
        if i <= 4: continue
        if i > 25: break
        # if i > 1: break

        vca_config = Config(f'{data_dir}/{file}', 0)
        vca_config.num_units = 40
        # vca_config.num_layers = 2
        vca_config.numsamples = 50
        vca_config.T0 = 1
        vca_config.lr = 5*(1e-4)
        vca_config.num_warmup_steps = 1000

        mean_energies, min_energies = run_vca(vca_config)

        with open(f"{results_dir}/VCA.txt", "a") as f:
            f.write(f"{min_energies}\n")
