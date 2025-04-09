import json
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import shutil

debugFlag = False
interactiveFlag = True

class XGBoost:
    def __init__(self, model=None):
        self.bst = None
        if model is not None:
            self.bst = xgb.Booster()
            self.bst.load_model(model)

    def plot_learning_curves(self, evals_result, feature_names):
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制训练集曲线
        ax.plot(evals_result['train']['logloss'],
                label='Training Error',
                color='blue')

        # 绘制验证集曲线
        ax.plot(evals_result['val']['logloss'],
                label='Validation Error',
                color='red')

        ax.set_xlabel('Boosting Rounds')
        ax.set_ylabel('logloss')
        ax.set_title('XGBoost Learning Curves')
        ax.legend()
        plt.grid(True)
        save_path = "datas/fig/learning_curves"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_paths = os.path.join(save_path, f"learning_curves_{feature_names}.png")
        plt.savefig(full_paths, dpi=300)

    def plot_roc_curve(self, fpr, tpr, feature_names):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', marker='o')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        auc_score = auc(fpr, tpr)
        plt.text(0.6, 0.2, f'AUC = {auc_score:.3f}', fontsize=12)
        plt.grid(True)
        save_path = "datas/fig/roc_curve"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_paths = os.path.join(save_path, f"roc_curve_{feature_names} auc:{auc_score:.3f}.png")
        plt.savefig(full_paths, dpi=300)

    def plot_pr_scatter_with_thresholds(self, precision, recall, thresholds, type, name, figsize=(10, 8),
                                           scatter_size=100,
                                           annotate_step=1,
                                           decimals=2,
                                           text_spacing=0.02):
        # 绘制散点和线
        plt.figure(figsize=figsize)
        scatter = plt.scatter(recall, precision,
                            c=thresholds,
                            s=scatter_size,
                            cmap='viridis',
                            alpha=0.6)

        plt.plot(recall, precision, 'b-', alpha=0.3, label='PR curve')

        # 创建文本标注位置调整器
        from adjustText import adjust_text
        texts = []

        # 添加文本标注
        for i in range(0, len(thresholds), annotate_step):
            text = plt.text(recall[i], precision[i],
                        f'prec:{precision[i]:.{decimals}f},rec:{recall[i]:.{decimals}f},thresh:{thresholds[i]:.{decimals}f}',
                        fontsize=4)
            texts.append(text)

        # 自动调整文本位置避免重叠
        adjust_text(texts,
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                expand_points=(1.5, 1.5))

        plt.colorbar(scatter, label='Threshold')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Thresholds')
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_path = f"datas/fig/{type}_pr_curve"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_paths = os.path.join(save_path, f"pr_curve_{type}_{name}.png")
        plt.savefig(full_paths, dpi=500)

    def Train(self, goodData, badData, goodLabels, badLabels, feature_names=None,
              param={'max_depth': 3, 'eta': 0.02, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0},
              num_boost_round=250
        ):
        evals_result = {}

        print(f"goodLabels_good_percent: {1 - sum(goodLabels)/len(goodLabels)}")
        print(f"badLabels_good_percent: {1 - sum(badLabels)/len(badLabels)} badLabels_bad_percent: {sum(badLabels)/len(badLabels)}")
        goodData = np.array(goodData)
        badData = np.array(badData)
        X = np.vstack((goodData, badData))
        y = np.concatenate((goodLabels, badLabels))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80%训练，20%测试
        dtrain = xgb.DMatrix(X_train[:, :-1].astype(float), label=y_train)
        dtest = xgb.DMatrix(X_test[:, :-1].astype(float))
        dval = xgb.DMatrix(X_test[:, :-1].astype(float), label=y_test)
        self.bst = xgb.train(param, dtrain=dtrain, num_boost_round=num_boost_round, evals=[(dtrain, 'train'), (dval, 'val')], evals_result=evals_result)
        self.plot_learning_curves(evals_result, feature_names)

        y_pred = self.bst.predict(dtest)
        for i, (test, pred, x) in enumerate(zip(y_test, y_pred, X_test[:, :-1].astype(float))):
            print(f"{feature_names[0]}: {x[-2]:.3f} {feature_names[1]}: {x[-1]:.3f} Test: {test:.3f} Predict: {pred:.3f} error: {(pred - test):.3f} name: {X_test[i][-1]}")

        sudden_turn_threshold2 = 0.5
        y_pred_labels = (y_pred > sudden_turn_threshold2).astype(int)
        accuracy = accuracy_score(y_test, y_pred_labels)

        # 计算ROC曲线
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label= 1, drop_intermediate=False)
        self.plot_roc_curve(fpr, tpr, feature_names)

        # 计算train PR曲线
        precision_train, recall_train, thresholds_train =  metrics.precision_recall_curve(y_train, self.bst.predict(dtrain), pos_label= 1)
        while len(precision_train) > len(thresholds_train):
            thresholds_train = np.insert(thresholds_train, -1, 1.0)
        self.plot_pr_scatter_with_thresholds(precision_train, recall_train, thresholds_train, 'train', f'{feature_names}', annotate_step=1)

        # 计算test PR曲线
        precision_test, recall_test, thresholds_test =  metrics.precision_recall_curve(y_test, y_pred, pos_label= 1)
        while len(precision_test) > len(thresholds_test):
            thresholds_test = np.insert(thresholds_test, -1, 1.0)
        self.plot_pr_scatter_with_thresholds(precision_test, recall_test, thresholds_test, 'test', f'{feature_names} accuracy: {accuracy:.3f}', annotate_step=1)
        return accuracy

    def Save(self, fileName):
        self.bst.save_model(fileName)


def Verify(model):
    xgboost = XGBoost(model)
    newData = [
        [
            2.0,
            2.0,
            2.0,
            2.0,
            0.3
        ]
    ]
    dnew = xgb.DMatrix(newData)
    predictions = xgboost.bst.predict(dnew)
    print(f"Predictions for positive data: {predictions}")

def get_train_features(features, labels, names, feature_pair, i):
    train_features = []
    train_labels = []
    for (case_feature, label, name) in zip(features, labels, names):
        if len(case_feature['ego_lat_jerk']) <= 1:
            continue
        ego_steer_angle = case_feature['ego_steer_angle']
        ego_lat_jerk = case_feature['ego_lat_jerk']
        max_lat_jerk_idx, max_lat_jerk_val = max(enumerate(ego_lat_jerk[:-1]), key=lambda x:abs(x[1]))
        if (max_lat_jerk_idx -  2 * i < 0):
            continue
        feature0_1 = ego_steer_angle[max_lat_jerk_idx]
        feature0_2 = ego_steer_angle[max_lat_jerk_idx - i]
        feature0_3 = ego_steer_angle[max_lat_jerk_idx - i * 2]

        # feature0_0 = ego_steer_angle[max_lat_jerk_idx - i]
        # feature0_1 = ego_steer_angle[max_lat_jerk_idx]
        # feature0_2 = ego_steer_angle[max_lat_jerk_idx] - ego_steer_angle[max_lat_jerk_idx - i]
        # feature0_3 = (case_feature['ego_steer_angle_rate'][max_lat_jerk_idx] - case_feature['ego_steer_angle_rate'][max_lat_jerk_idx - i]) / 10.0
        # feature1 = case_feature[feature_pair[0]][max_lat_jerk_idx]
        # feature2 = case_feature[feature_pair[1]][max_lat_jerk_idx]
        train_feature = [feature0_1, feature0_2, feature0_3, name]
        train_features.append(train_feature)
        train_labels.append(label)
    return train_features, train_labels

def clear_directory(directory):
    """清空目录下的所有文件，保留目录结构"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        print(f"deleting {file_path}")
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'删除失败 {file_path}: {e}')