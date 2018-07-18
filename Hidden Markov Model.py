import numpy as np
from hmmlearn import hmm
np.random.seed(42)

modelmm = hmm.GaussianHMM(n_components=7, covariance_type="full")
modelmm.startprob_ = np.array([0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19])
modelmm.transmat_ = np.array([[0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.11, 0.21, 0.21, 0.21, 0.21],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19]])
modelmm.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
modelmm.covars_ = np.tile(np.identity(2), (3, 1,1))
X, Z = modelmm.sample(100)


remodel = hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=100)
X_markov = entireData3.reshape(entireData3.shape[0], entireData3.shape[1])
X_markov2 = pd.DataFrame(X_markov)
X_markov2 = X_markov2.values.flatten()
lengths =[]
#for j in range(np.size(X_markov,0))
for i in range(len(X_markov)):
    lengths.append(len(X_markov[i]))
#fit para todos os alunos
remodel.fit(np.reshape(X_markov2,(len(X_markov2),1),lengths))
hmm.GaussianHMM(n_components=5).fit(X_markov2, lengths)

#fit para 1 aluno 
remodel.fit(np.reshape(X_markov[0],(np.size(X_markov,1),1),lengths))

X_submit_markov = np.reshape(X_submit,(len(X_submit),40))
X_submit_markov = pd.DataFrame(X_submit_markov)
X_submit_markov = X_submit_markov.values.flatten()

#q1 = np.reshape(X_submit_markov[0],(np.size(X_submit,1),1))
aaa = remodel.predict(np.reshape(X_submit_markov,(len(X_submit_markov),1),lengths[:len(X_submit)]))#lengths[:len(X_submit)]
bbb = np.reshape(aaa,(len(X_submit),40))


question_final = pd.DataFrame(bbb[:,:5],columns=['Q41','Q42','Q43','Q44','Q45'])
question_final[question_final == 0] = 'A'
question_final[question_final == 1] = 'B'
question_final[question_final == 2] = 'C'
question_final[question_final == 3] = 'D'
question_final[question_final == 4] = 'E'
question_final = question_final.apply(lambda x: dcat[x.name].inverse_transform(x))



