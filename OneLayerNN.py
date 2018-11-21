import numpy as np

# %% learn with proximal gradient descent
'''
softmax element wise
SM(beta_i)= beta_i  - lamda*eta    if lamda*eta < beta_i
SM(beta_i) = 0                      if  -lamda*eta<beta_i <lamda*eta <
SM(beta_i) = beta_i  - lamda*eta    if -lamda*eta > beta_i
minimize 1/2|| y - x*beta||^2 + lamda||beta||
if our g(x) term is 1/2|| y - x*beta||^2
then our h(x)term lamda||beta|| term

'''
step_size = 0.5
lambd = 1

def softmax(w,stepSize,Lambda):
    print('original',w)
    update_rule = stepSize * Lambda
    for idx, val in enumerate(w):
        print(val)
        print(idx)
        if update_rule < val:
            w[idx] = w[idx] - update_rule
        elif update_rule < -val:
            w[idx] = w[idx] + update_rule
        else :
            w[idx] = 0

    print("new w",w)
    return(w)

train_x = np.array([[1, 2 ,3 ,4 ,5],[6,7,8,9,0]])
train_y = np.array([[0],[1],[1],[1],[1]])
# x train is in the for m samples by n features. m rows n columns
train_x = np.transpose(train_x)
#%%
# initialize beta as a vector of all zeros
beta = np.zeros((np.shape(train_x)[1], 1))
beta.shape

# %%
print('shape xtrain', np.shape(train_x))
print('shape beta', np.shape(beta))
print('shape y_trian',np.shape( train_y))

y_pred = np.dot(train_x, beta)
print(y_pred)
# %%

X_T = np.transpose(train_x)

print("Finding gradient...")
grad_g = (-1) * np.dot(X_T, train_y - y_pred)
'''
gradient_g=-X^T(y-X*beta_t)

update_beta_t_1_2 = beta + step_size* gradient_g

update_beta_t_1 = proximal(beta_t+step_size*X^T*(y-X*beta_t))
                = softMax_Eq( beta_t + step_size*X^T*(y-X*beta_t))
'''

print("Computing beta value before softmax ...")
update_beta = beta + step_size * grad_g

print("Applying softmax ...")
new_beta = softmax(update_beta,step_size,lambd)