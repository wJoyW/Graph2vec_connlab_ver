# Graph2vec_connlab_ver
對生成WL graph之解說
1. 利用WL algo將function name與hash後之label組成一新features
2. 利用doc2vec將上述feature轉換成向量，以利於機器學習

好處：向量feature較於輕便，相較於蒐集label個數與儲存，且機器學習可在短時間內完成訓練
壞處：在doc2vec環節較於耗時，經測試生成向量feature平均約780秒

--------------------------------------------------------------------------------

To construct a WL graph：
1. Concatenating the function name and label after hashing and using WL algo. to make new features.
2.Using doc2vec to transform the features above to vectors. These vectors will be the input for our machine learning model.

advantage: 
The feature is less sophisticate to store compares to store pure label's quantities.
It can be trained in less time during machine leaening.

disadvantage:
The time spends in doc2vec is time consuming. The experiment shows that the average of
making vectors by features spends 780 seconds.
