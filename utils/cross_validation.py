import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
# import matplotlib
# matplotlib.use('TkAgg')
pd.set_option('display.expand_frame_repr', False)


random.seed(1211)


def my_k_fold(df, k_fold=5):
    indexes = list(range(len(df)))
    random.shuffle(indexes)
    list_k_fold_splits = []
    for k in range(0, k_fold):
        fract = int(np.floor(len(indexes)/k_fold))
        ts_idx = indexes[fract * k : (fract * k) + fract]
        tr_idx = [i for i in indexes if i not in ts_idx]
        list_k_fold_splits.append((tr_idx, ts_idx))
    return list_k_fold_splits

def my_leave_one_out(df):
    return my_k_fold(df, k_fold=len(df))


if __name__ == "__main__":
    df = pd.read_csv('winequality-red.csv', sep=';')

    list_k_fold_splits = my_k_fold(df, k_fold=5)
    print(df.info())
    # dop.drop(columns='DRUPE_COLOR', inplace=True)

    ####################################################################################################################
    #                                       CROSS-VALIDATION FOR MODEL ASSESSMENT                                      #
    ####################################################################################################################
    f1_cv_ASSESSMENT = []
    # Assessment
    kth_fold = 1
    splits = list_k_fold_splits
    for tr_idx, ts_idx in splits:
        print("Working on split " + str(kth_fold))
        dop_tr = df.iloc[tr_idx]
        dop_ts = df.iloc[ts_idx]
        f, axes = plt.subplots(2, 1)
        sns.countplot(x='quality', data= dop_tr, orient='h', ax=axes[0])
        sns.countplot(x='quality', data= dop_ts, orient='h', ax=axes[1])
        axes[0].set_title('Training split')
        axes[1].set_title('Test split')
        f.suptitle('Split' + str(kth_fold))
        f.show()
        #plt.show()
        kth_fold +=1

        y_tr = dop_tr.quality.values # dop_tr['quality'].values
        x_tr = dop_tr.drop(columns='quality')

        current_knn = KNeighborsClassifier(n_neighbors=5)
        current_knn.fit(x_tr, y_tr)

        y_ts = dop_ts.quality.values
        x_ts = dop_ts.drop(columns='quality')
        y_pred = current_knn.predict(x_ts)

        df_test = pd.DataFrame({'Actual': y_ts, 'Predicted': y_pred})
        df_test = df_test[0:20]
        plt.figure(1)
        df_test.plot(kind='bar', figsize=(10, 8))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
        #plt.show()
        plt.figure(2)
        cm = confusion_matrix(y_ts, y_pred)
        sns.heatmap(cm, annot=True)
        #plt.show()

        curr_score = f1_score(y_ts, y_pred, average='macro')
        f1_cv_ASSESSMENT.append(curr_score)


## Report
    print('The cross-validated F1-score of your algorithm is ', np.mean(np.asarray(f1_cv_ASSESSMENT)))
#
#     ####################################################################################################################
#     #                                       CROSS-VALIDATION FOR MODEL SELECTION                                       #
#     ####################################################################################################################
    df_train = df[0:int(0.7*len(df))]
    list_k_fold_splits = my_k_fold(df_train, k_fold=5)
    f1_cv_SEL = []
    # Define parameter search space
    k_parameter = range(1, 8, 2)
    # Selection
    for k_par in k_parameter:
        f1_cv_par = []
        for tr_idx, ts_idx in splits:
            print("Working on split " + str(kth_fold))
            dop_tr = df.iloc[tr_idx]
            dop_ts = df.iloc[ts_idx]
            kth_fold += 1
            y_tr = dop_tr.quality.values  # dop_tr['quality'].values
            x_tr = dop_tr.drop(columns='quality')
            current_knn = KNeighborsClassifier(n_neighbors=k_par)
            current_knn.fit(x_tr, y_tr)
            y_ts = dop_ts.quality.values
            x_ts = dop_ts.drop(columns='quality')
            y_pred = current_knn.predict(x_ts)
            curr_score = f1_score(y_ts, y_pred, average='macro')
            f1_cv_par.append(curr_score)
        f1_cv_SEL.append(np.mean(np.asarray(f1_cv_par)))

    # Report
    print('The cross-validated F1-scores of your algorithm with the explored parameters are: ')
    for i in range(len(k_parameter)):
        print('For k = ', k_parameter[i], ' --> F1-score = ', f1_cv_SEL[i])
    print('Overall, the best value for parameter k is ', k_parameter[np.argmax(np.asarray(f1_cv_SEL))],
          ' since it leads to F1-score = ', f1_cv_SEL[np.argmax(np.asarray(f1_cv_SEL))])

#
# ########################################################################################################################
# # Uhm... Actually! Sklearn already has the functions we implemented.
# ########################################################################################################################
#
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
splits = KFold(n_splits=5, shuffle=True).split(dop)
#