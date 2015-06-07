import numpy as np
from transfer_funcs import *;
from dataset_mgmt import *;


transfer = sigmoid_transfer; transfer_deriv = sigmoid_transfer_deriv; cost_deriv = cost_MSE_deriv;
def nnet_setup(node_layout, transferp=sigmoid_transfer, 
               transfer_derivp=sigmoid_transfer_deriv, cost_derivp=cost_MSE_deriv):
    global transfer; global transfer_deriv; global cost_deriv;
    transfer = transferp; transfer_deriv = transfer_derivp; cost_deriv = cost_derivp;
    
    weights = []; biases = []
    for i in range(1, len(node_layout)):
        weights.append(np.random.randn(node_layout[i], node_layout[i-1]))
        biases.append(np.random.randn(node_layout[i], 1))
    return np.array(weights), np.array(biases)

# Basic forward propagation
def nnet_prop(weights, biases, inputs):
    assert(len(np.array(inputs).shape) == 2)
    for w, b in zip(weights, biases):
        inputs = transfer(np.dot(w, inputs) + b)
    return inputs

def nnet_backpropagate(inp, outp, wts, bias, outp_length=10):
    del_w = [np.zeros(shape=wt.shape) for wt in wts]
    del_b = [np.zeros(shape=bt.shape) for bt in bias]
    
    next_input = conv_to_col(inp)
    outp = create_tgt_vec(outp, length=outp_length)
    
    pre_trans = []; post_trans = []
    for w, b in zip(wts, bias):
        next_input = np.dot(w, next_input) + b
        pre_trans.append(next_input)
        next_input = transfer(next_input)
        post_trans.append(next_input)
    
    #print("O: ", outp)
    delta = cost_deriv(post_trans[-1], outp) * transfer_deriv(pre_trans[-1])
    del_b[-1] = delta
    del_w[-1] = np.dot(delta, post_trans[-2].transpose())
    
    for i in range(2, len(wts)):
        pre_tr_vec = pre_trans[-i]
        tr_deriv = transfer_deriv(pre_tr_vec)
        delta = np.dot(wts[-i+1].transpose(), delta) * tr_deriv
        del_b[-i] = delta
        del_w[-i] = np.dot(delta, post_trans[-i-1].transpose())
    
    return del_w, del_b

def nnet_SGD(train_set, wts, bias, eta, backprop_fn=nnet_backpropagate, outp_length=10, decay_rate=0.0):
    training_size = 0
    if len(train_set) == 0: training_size = 1
    else: training_size = len(train_set[0])

    learning_coef = eta / training_size
    
    for next_set in train_set:
        sum_del_w = [np.zeros(w.shape) for w in wts]
        sum_del_b = [np.zeros(b.shape) for b in bias]
        
        for inp, outp in next_set:
            next_del_w, next_del_b = backprop_fn(inp, outp, wts, bias, outp_length=outp_length);
            sum_del_w = [nw + dw for nw, dw in zip(next_del_w, sum_del_w)];
            sum_del_b = [nb + db for nb, db in zip(next_del_b, sum_del_b)];
        
        wts  = [wt - learning_coef * (dw + decay_rate * wt) for wt, dw in zip(wts, sum_del_w)]
        bias = [bt - learning_coef * (db + decay_rate * bt) for bt, db in zip(bias, sum_del_b)]
    return wts, bias

def train_new(networkQuantity, networkKeepQuantity, networkEta,
             networkDecayRate, firstLayer, secondLayer,
             trainingEpochs, trainingSetSize, trainingBatchSize):
    networks = []
    _, test_set = load_data(trainingSetSize, trainingBatchSize)
    for i in range(0, networkQuantity):
        if secondLayer == 0 or secondLayer == None: wts, bias = nnet_setup([64, firstLayer, 10])
        else: wts, bias = nnet_setup([64, firstLayer, secondLayer, 10])

        wts_maxA = wts; bias_maxA = bias; effectivenessA = nnet_evaluate_single(wts, bias, test_set)[2]
        for j in range(0, trainingEpochs):
            train_set_split, test_set = load_data(trainingSetSize, trainingBatchSize)
            wts, bias = nnet_SGD(train_set_split, wts, bias, networkEta, decay_rate=networkDecayRate)

            new_effect = nnet_evaluate_single(wts, bias, test_set)[2]
            if new_effect > effectivenessA:
                effectivenessA = new_effect
                wts_maxA = wts; bias_maxA = bias

        #if len(networks) > 0: print("Low: " + str(networks[0][2]), "New: " + str(effectivenessA))
        if len(networks) < networkKeepQuantity:
            #print("Adding new network to unfinished list")
            networks.append((wts_maxA, bias_maxA, effectivenessA))
        else: 
            if networks[0][2] < effectivenessA:
                # print("Dumping old network for new one")
                networks[0] = (wts_maxA, bias_maxA, effectivenessA)
        networks = sorted(networks, key=lambda x: x[2])
        # print("Current accuracy: ", nnet_evaluate_multiple(networks, test_set), "\n")
    return networks

def nnet_evaluate_single(wts, bias, test_set):
    correct = 0; total = 0;
    for i in test_set:
        out = read_outp_vec(nnet_prop(wts, bias, conv_to_col(i[0])))
        total = total + 1
        if out == i[1]: correct = correct + 1
    return correct, total, float(correct)/float(total)

def nnet_evaluate_multiple(networks, test_set):
    correct = 0; total = 0;
    for i in test_set:
        out = np.zeros(shape=create_tgt_vec(0).shape)
        sumEff = 0
        for wts, bias, efA in networks:
            out = out + efA * nnet_prop(wts, bias, conv_to_col(i[0]))
            sumEff += efA
        out = out / sumEff
        
        if read_outp_vec(out) == i[1]: correct = correct + 1
        total = total + 1
    return correct, total, float(correct)/float(total)

### ASSERT TESTS ###

# NNet setup shaping
assert(nnet_setup([1, 2, 1])[0][0].shape == (2, 1))
assert(nnet_setup([1, 2, 1])[0][1].shape == (1, 2))
assert(nnet_setup([1, 2, 1])[1][0].shape == (2, 1))
assert(nnet_setup([1, 2, 1])[1][1].shape == (1, 1))

# Basic feedforward testing
wts, bias = nnet_setup([1, 2, 1], linear_transfer, linear_transfer_deriv)
for i in wts: 
    for j in i: j.fill(1)
for i in bias: 
    for j in i: j.fill(0)
out = nnet_prop(wts, bias, [[1]])
assert(len(out.shape) == 2)
assert(out[0][0] == 2)

for i in wts:
    for j in i: j.fill(1)
for i in bias:
    for j in i: j.fill(-1)

assert(nnet_prop(wts, bias, [[1]]) == -1)

# SGD testing
def mock_backprop(inp, outp, wts, bias, outp_length=10):
    return wts, bias

wts, bias = nnet_setup([1, 2, 1])
for i in wts:
    for j in i:
        j.fill(1)
for i in bias:
    i.fill(1)

for i in range(0, 1000):
    wts, bias = nnet_SGD([[(1, 1)]], wts, bias, 0.1, backprop_fn=mock_backprop)
for i in wts:
    for j in i:
        assert(np.allclose(j, 0, 0.001))
for i in bias:
    assert(np.allclose(i, 0, 0.001))