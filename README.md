# All-About-Pycaret

![](https://github.com/pycaret/pycaret/blob/master/docs/images/logo.png?raw=true)
**An open-source, Low-code machine learning library in Python**

**🚀 PyCaret 3.0-rc is now out.**

오픈 소스이자, 코드가 많이 필요하지 않은 머신러닝 라이브러리인 **Pycaret**에 대한 스터디입니다.

![](https://github.com/pycaret/pycaret/raw/master/docs/images/quick_start.gif)

------------------------------
## NoteBook List

| Dataset  | Task                | # of Instances | # of Features | Notebook |
|----------|---------------------|----------------:|---------------:|----------|
| anomaly  | Anomaly Detection   | 1000           | 10            | [link]() |
| titanic  | Classification      | 891            | 11            | [link]() |
| iris     | Classification      | 150            | 5             | [link]() |
| house    | Regression          | 1461           | 81            | [link]() |
| kiva     | NLP/ Classification | 6818           | 7             | [link]() |
| facebook | Clustering          | 7050           | 12            | [link]() |
| tweets   | NLP                 | 8594           | 2             | [link]() |


## Pycaret GPU Support
- Extreme Gradient Boosting (Without Installation)
- CatBoost (Without Installation)
- Light Gradient Boosting Machine with GPU installation
- XGBoost with XGBoost Library Installation
-------------------------------
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Pycaret 3.0.

```bash
pip install pycaret
```

```bash
pip install pycaret[full]
```

-----------------------------
## Usage
### Supervised Workflow
#### Classification 
```python
import pandas as pd
from pycaret.classification import *

# load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# init setup
s = setup(train, target = 'target')

# model training and selection
best = compare_models()

# analyze best model
evaluate_model(best)

# predict on new data
predictions = predict_model(best, data = test)

# save best pipeline
save_model(best, 'best_model')
```

#### Regression
```python
import pandas as pd
from pycaret.regression import *

# load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# init setup
s = setup(train, target = 'target')

# model training and selection
best = compare_models()

# analyze best model
evaluate_model(best)

# predict on new data
predictions = predict_model(best, data = test)

# save best pipeline
save_model(best, 'best_model')
```


### Unsupervised Workflow
#### Clustering
```python
import pandas as pd
from pycaret.clustering import *

# load dataset
data = pd.read_csv('data.csv')

# init setup
s = setup(data, normalize=True)

# train K-means model
best = creat_model('kmeans')

# assign cluster labels on training data
kmeans_results = assign_model(best)

# assign cluster labels on enw data
new_data = pd.read_csv('new_data.csv')
predictions = predict_model(best, data=new_data)

# save best pipeline
save_model(best, 'best_model')
```

#### Anomaly Detection
```python
import pandas as pd
from pycaret.anomaly import *

# load dataset
data = pd.read_csv('data.csv')

# init setup
s = setup(train, normalize = True)

# train isolation forest model
best = create_model('iforest')

# assign anomaly labels on training data
iforest_results = assign_model(iforest)

# assign anomaly labels on new data
new_data = pd.read_csv('new_data.csv')
predictions = predict_model(best, data=new_data)

# save best pipeline
save_model(best, 'best_model')
```


## Contributing
@[jhbale11](https://github.com/jhbale11?tab=repositories)

## License
[MIT](https://choosealicense.com/licenses/mit/)
