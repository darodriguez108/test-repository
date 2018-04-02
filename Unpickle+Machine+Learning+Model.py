
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import os
import pandas as pd


# In[ ]:


pre = os.path.dirname(os.path.realpath('__file__'))
fname = 'DataReadyforKNNModel.xlsx'
path = os.path.join(pre, fname)
data = pd.read_excel(path)


# In[ ]:


data.head()


# In[ ]:


sdm = data['Sabot Depth'].mean()
sds = data['Sabot Depth'].std()

mm = data['Muzzle'].mean()
ms = data['Muzzle'].std()

som = data['Sabot_OD'].mean()
sos = data['Sabot_OD'].std()

avm = data['Actual_Velocity'].mean()
avs = data['Actual_Velocity'].std()

lpm = data['Lp_Mass'].mean()
lps = data['Lp_Mass'].std()

sm = data['Squeeze'].mean()
ss = data['Squeeze'].std()


# In[ ]:


print((7-sdm)/(sds))
print((0.4965-mm)/(ms))
print((0.4975-som)/(sos))
print((6.94-avm)/(avs))
print((1.1898-lpm)/(lps))
print((0.001-sm)/(ss))


# In[ ]:


knn_pkl = open("KNNRegressionModel.pkl","rb")


# In[ ]:


model = pickle.load(knn_pkl)


# In[ ]:


model


# In[ ]:


#put inputs (sabot depth, muzzle, sabot OD, Velocity, LP Mass, Squeeze)
a = float(input("Please Enter Sabot Depth: "))
b = float(input("Please Enter Muzzle: "))
c = float(input("Please Enter sabot OD: "))
d = float(input("Please Enter Velocity: "))
e = float(input("Please Enter LP Mass: "))
f = c-b

ab = (a-sdm)/(sds)
bb = (b-mm)/(ms)
cb = (c-som)/(sos)
db = (d-avm)/(avs)
eb = (e-lpm)/(lps)
fb = (f-sm)/(ss)


sabotd, muzzle, sabotOD, Velocity, LPMass, Squeeze = ab, bb, cb, db, eb, fb   
inputs = np.array([[sabotd,muzzle,sabotOD,Velocity,LPMass,Squeeze]])

print("\n\n Please use a powder load of : \n")
model.predict(inputs)

