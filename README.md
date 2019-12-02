#  四、项目运行办法
# **项目的文件结构**
person-reid-2019-NAIC：  
│    &emsp;&emsp;&ensp;BDB.json  
│    &emsp;&emsp;&ensp;ensemble.py  
│    &emsp;&emsp;&ensp;final_battle.json  
│    &emsp;&emsp;&ensp;main.py  
│    &emsp;&emsp;&ensp;main_BDB.py  
│    &emsp;&emsp;&ensp;mgn.json  
│    &emsp;&emsp;&ensp;option.py  
│    &emsp;&emsp;&ensp;README.md  
│    &emsp;&emsp;&ensp;trainer.py  
│    
├─data  
│        &emsp;&emsp;&ensp;common.py  
│        &emsp;&emsp;&ensp;contest.py  
│        &emsp;&emsp;&ensp;sampler.py  
│        &emsp;&emsp;&ensp;__init__.py  
│        
├─loss  
│        &emsp;&emsp;&ensp;AngularSoftmaxWithLoss.py  
│        &emsp;&emsp;&ensp;CrossEntropyLabelSmooth.py  
│        &emsp;&emsp;&ensp;metric.py  
│        &emsp;&emsp;&ensp;triplet.py  
│        &emsp;&emsp;&ensp;TripletLoss.py  
│        &emsp;&emsp;&ensp;__init__.py  
│        
├─model  
│        &emsp;&emsp;&ensp;mgn.py  
│        &emsp;&emsp;&ensp;__init__.py  
│        
└─utils  
          &emsp;&emsp;&ensp;functions.py  
          &emsp;&emsp;&ensp;nadam.py  
          &emsp;&emsp;&ensp;n_adam.py  
          &emsp;&emsp;&ensp;random_erasing.py  
          &emsp;&emsp;&ensp;re_ranking.py  
          &emsp;&emsp;&ensp;utility.py  
# **项目的运行步骤**
1.运行main.py  
2.运行ensemble.py（确保BDB.json与mgn.json成功生成并放在./home下）  
3.在final_battle.json文件夹中含有生成的结果  
# **运行结果的位置**
1.final_battle.json 文件
