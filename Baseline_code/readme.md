


## Possible Error Fixes Required While Running the Baseline Code

If you encounter issues while running the baseline code, try the following fixes:

### 1. Fix `greenlet` version  
```bash
pip uninstall greenlet -y 
pip install greenlet==0.4.17
```

### 2. Fix `overrides` version  
```bash
pip uninstall overrides -y 
pip install overrides==3.1.0
```

### 3. Train using CPU ( GPU Resource Constraints )
```bash
allennlp train -f --include-package diplomacy -s logdir --overrides '{"trainer": {"cuda_device": -1}}' configs/actual_lie/contextlstm+power.jsonnet
```

### 4. Install required dependencies  
```bash
pip install pandas seaborn wordcloud
```

