# Two step principal component analysis implementation.
Implementation of two step principal component analysis based process monitoring technique developed by Zhijiang Lou; Jianyong Tuo; Youqing Wang. 
The algorithm and all other theory is thoroughly described in the following publicashion: https://ieeexplore.ieee.org/document/7983758

# Example usage
two_step_pca.py exposes TS_PCA class, which contains model's parameters, method to fit the parameters and a method to detect 
anomalies in a timeseries data. 

Fitting the model: 

Fitting method accepts pandas dataframe or numpy array as data input, where columns are separate features and rows are measurements'
values for each timestamp; max lag parameter; max diference value; verbosity level. You can find the meaning of the parameters from the 
linked paper.


```python
import two_step_pca as tsp
import pandas as pd # you need either pandas or numpy to keep the data for fitting
import numpy as np
import pickle

ts_pca = tsp.TS_PCA() # initialize the model
ts_pca.fit(dataframe, qMAX, dMAX, verbose=True) 
with open("model", "wb") as output:
    pickle.dump(ts_pca, output, pickle.HIGHEST_PROTOCOL) # save trained model for later usage
```

Detecting anomalies in the data:
Detection method accepts data (as numpy array or pandas dataframe); explained variation (determines how many principal
components are used during detection; keep it around 0.8 - 0.9 as a rule of thumb). The method returns pandas dataframe with Hotelling's T2
and SPE metrics, which show how "suspicious" the data is.

```python
import two_step_pca as tsp
import pandas as pd # you need pandas or numpy for passing data for detection and dataframe for recieving output from detection
import numpy as np

metrics = ts_pca.detect(dataframe, var_explained)
# metrics has two columns "SPE" and "T2".
```
