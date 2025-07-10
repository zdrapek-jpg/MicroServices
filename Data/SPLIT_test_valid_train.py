import random

import numpy as np
import pandas as pd

class SplitData:
    __dict__ = ["train","valid","test","groups"]
    """
        A class to split a dataset into training, validation, and test sets.
        Parameters:
            train (float):  Default is 0.4.
            valid (float):  Default is 0.4.
            test (float):  Default is 0.2.
        The sum of train, valid, and test must be 1.0."""
    train = 0.4
    valid = 0.4
    test = 0.2
    groups =2
    @classmethod
    def set(cls,train=0.6,valid=0.2,test=0.2,groups=2):
        cls.train = train  # 0.40
        cls.valid = valid  # 0.40
        cls.test = test  # 0.20
        cls.groups=groups
    @classmethod
    def split_in_groups(cls,x,groups=None,size=0.7):
        """
        :param x: np.array data n feauters, f samples
        :param y:  zmienna opisująca value
        :param groups: number of groups for split to get indexes of data
        :return: valid_index, train indexes
        """
        if groups!=None:
            cls.groups=groups
        from sklearn.model_selection import ShuffleSplit
        groups_kFould =ShuffleSplit(n_splits=cls.groups,train_size=size)


        return list(groups_kFould.split(x))
    def get_in_order(self,list_of_kfolds:list=None):
        kfold_order = list( range(0,len(list_of_kfolds)))
        random.shuffle(kfold_order)
        return kfold_order

    @classmethod
    def split_data(cls,data:pd.core.frame.DataFrame)->np.array:
        """data must be in data frame where [ x] :
        and last column is  y
        data must be normalized bofor splitting
        basic setting is train = 0.4 valid =0 .4 and test=0.2"""
        if cls.train+cls.valid+cls.test<1:
            cls.train+= 1 - (cls.valid+cls.test)
        if  0>cls.train+cls.valid+cls.test>1:
            raise "nie można podzielić zbioru podano podział który uwzględdnia ponad 100 % zbioru"
        shuffled_data= data.sample(frac=1,random_state=43).reset_index(drop=True)


        ## augumentation, kkrotna walidacja skrośna,crossvalidation!!
        # cał zbiór jest dielony na ustaloną ilość,
        # trening odbywa się że z 5 jest wybierany jeden zbiór który jest wybierany na vaildacyjny
        #valid =5 train[1,4] i iterujemy się po koleji zmieniając grupe walidacyjną
        #

        # podzielenie na x i y
        # domyślnie że y jest ostanią kolumną
        len_of_data = shuffled_data.shape[0]
        split1 = int(len_of_data*cls.train)
        split2 = int(len_of_data*cls.valid)+split1
        shuffled_data.rename(columns={shuffled_data.columns[-1]: "y"}, inplace=True)

        x_train,y_train = shuffled_data.iloc[:split1,:-1].values,       shuffled_data.iloc[:split1,-1].values
        x_valid,y_valid = shuffled_data.iloc[split1:split2,:-1].values, shuffled_data.iloc[split1:split2,-1].values
        x_test,y_test = shuffled_data.iloc[split2:,:-1].values,         shuffled_data.iloc[split2:,-1].values
        return x_train,y_train,x_valid,y_valid,x_test,y_test

    @classmethod
    def tasowanie(cls,data,f= False):
        """ :param data is pd.DataFrame object
            :param f  when is set to False it splits data for x and y
            :param f  is set to True it returns pd.DataFrame"""
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        if f:  return shuffled_data
        return shuffled_data.iloc[:, :-1].values, shuffled_data.loc[:, "y"]
    @classmethod
    def merge(cls,x,y)->pd.core.frame.DataFrame:
        """:return x and y merged into one dataframe with last colum named y"""
        assert x.shape[0]==y.shape[0]," obiekty nie mają tego samego rozmiaru"
        data = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
        data["y"] = y
        return data

    @classmethod
    def batch_split_data(cls,data_frame,size:int)->tuple[list,list]:
        """
          :param data_frame is pd.DataFrame
          :param size defines size of every batch

          function shuffle data and cretates batches in list of x nad y
          :return list of batches  x,y
          """
        shuffled_data = SplitData.tasowanie(data_frame,f=True)

        # ile jest elementów
        len_data =shuffled_data.shape[0]
        # ile na 1 batch
        podzial = int(len_data // size)
        #tworzymy mini batche

        start, stop = 0, podzial
        batch_x, batch_y = [], []
        for i in range(size):
            batch_x.append(data_frame.iloc[start:stop, :-1].values)
            batch_y.append(data_frame.iloc[start:stop,-1].values)
            start += podzial
            stop += podzial
        batch_x.append(data_frame.iloc[start:,:-1].values)
        batch_y.append(data_frame.iloc[start:,-1].values)
        return batch_x,batch_y


