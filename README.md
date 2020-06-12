# Diverse decision sets

To run all methods on the example dataset, Iris:

```
$ python run.py 1 True iris
```

Some dependent libraries for baselines:

```
$ pip install pyarc # For CBA
$ conda install orange3 # For CN2
```

IDS is adapted from [here](https://github.com/lvhimabindu/interpretable_decision_sets).
Their code is too old to run and has bugs.
I fixed them.

Project organization:

 * Data is preprocessed in data-multi.ipynb
 * Metrics are generated in table.ipynb. However you need Sacred to manage the experimental data.


In case you wonder, this repo adopts [sacred](https://github.com/IDSIA/sacred) and [incense](https://github.com/JarnoRFB/incense) to manage experiment results.
However, you don't need to follow the same routine.
Without Sacred, results will be redirected to stdout.

```
$ pip install dnspython incense sacred 

# a private file: mongodburi.py
mongo_uri = 'mongodb+srv://xxx'
db_name = 'yyy'
```

This repo also uses Python type checking feature mainly for annotation and `pytest` for testing.

```
pip install pytest 

$ pytest
```


Datasets can be downloaded from the following list.

* http://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29
* http://archive.ics.uci.edu/ml/datasets/Avila
* http://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
* http://archive.ics.uci.edu/ml/datasets/Cardiotocography
* http://archive.ics.uci.edu/ml/datasets/Poker+Hand

