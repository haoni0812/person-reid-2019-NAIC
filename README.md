#  四、项目运行办法
# **项目的文件结构**
person-reid-2019-NAIC：
│  BDB.json
│  ensemble.py
│  final_battle.json
│  main.py
│  main_BDB.py
│  mgn.json
│  option.py
│  README.md
│  trainer.py
│  
├─data
│      common.py
│      contest.py
│      sampler.py
│      __init__.py
│      
├─loss
│      AngularSoftmaxWithLoss.py
│      CrossEntropyLabelSmooth.py
│      metric.py
│      triplet.py
│      TripletLoss.py
│      __init__.py
│      
├─model
│      mgn.py
│      __init__.py
│      
└─utils
        functions.py
        nadam.py
        n_adam.py
        random_erasing.py
        re_ranking.py
        utility.py
# **项目的运行步骤**
1.运行main.py
2.运行ensemble.py（确保BDB.json与mgn.json成功生成并放在./home下）
3.在final_battle.json文件夹中含有生成的结果
# **运行结果的位置**
1.final_battle.json 文件
