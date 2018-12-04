from sklearn.svm import SVC
from mnist import MNIST
from sklearn.metrics import accuracy_score
import numpy as np
'''Example of format


model = SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
 class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
model.fit(X,Y)
y_hat = model.predict(X_test)
'''
#%%

mndataSet = '.\Dataset'
mndata = MNIST()
#mndata.gz =True
mndata = MNIST(mndataSet)
#%%importing data into lists and labels
print("loading images....")
images_train,labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()
#normalize test
images_test = np.array(images_test)/255.0
labels_test = np.array(labels_test)

#%%
xtrain = np.array(images_train)/255.0
xlabels = np.array(labels_train)
print('building....')
c = [.25,.5,1,2,4]
kern = 'rbf'
model = SVC(C = 5,gamma = 0.001, kernel = kern)
model.fit(xtrain, xlabels)
y_hat = model.predict(xtrain)

#%%
accuracy  = accuracy_score(np.array(xlabels),y_hat)

print("Training accuracy ",accuracy)
'''
misClass = 0
for x in range(0,len(y_hat)):
    if x%15 == 0:
        print(y_hat[x])
        print(xlabels[x])
    if y_hat[x] == xlabels[x]:
        print('ok')
    else:
        misClass = misClass+1

train_error = misClass/len(y_hat)
print(train_error)
'''

#%%
print("Running Test...")
y_hat_test =  model.predict(images_test)
accuracyTest= accuracy_score(labels_test, y_hat_test)
print("Test accuracy " , accuracyTest)

fileSave='SavedModels'+os.sep+'SVC_savedModel_'+kern+'_'+str(datetime.timestamp(datetime.now())).replace('.','')+'.sav'
pickle.dump(model,open(fileSave,'wb'))