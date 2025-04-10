Possible Error Fixes required while running the baseline code 

pip uninstall greenlet -y 
pip install greenlet==0.4.17 
pip uninstall overrides -y 
pip install overrides==3.1.0
allennlp train -f --include-package diplomacy -s logdir --overrides '{"trainer": {"cuda_device": -1}}' configs/actual_lie/contextlstm.jsonnet
pip install pandas
pip install seaborn
pip install wordcloud

