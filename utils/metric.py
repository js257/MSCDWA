import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score

def score_model(preds, labels, use_zero=False):
    mae = np.mean(np.absolute(preds - labels))
    corr = np.corrcoef(preds, labels)[0][1]
    non_zeros = np.array(
        [i for i, e in enumerate(labels) if e != 0 or use_zero])
    preds = preds[non_zeros]
    labels = labels[non_zeros]
    preds = preds >= 0
    labels = labels >= 0
    f_score = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return acc, mae, corr, f_score

###SIMS數據集##

def __multiclass_acc(y_pred, y_true):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

def eval_sims_regression(y_pred, y_true):
    # test_preds = y_pred.view(-1).cpu().detach().numpy()
    # test_truth = y_pred
    test_preds = y_pred
    test_truth = y_true
    test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

    # weak sentiment two classes{[-0.6, 0.0], (0.0, 0.6]}
    ms_2 = [-1.01, 0.0, 1.01]
    weak_index_l = np.where(test_truth >= -0.4)[0]
    weak_index_r = np.where(test_truth <= 0.4)[0]
    weak_index = [x for x in weak_index_l if x in weak_index_r]
    test_preds_weak = test_preds[weak_index]
    test_truth_weak = test_truth[weak_index]
    test_preds_a2_weak = test_preds_weak.copy()
    test_truth_a2_weak = test_truth_weak.copy()
    for i in range(2):
        test_preds_a2_weak[np.logical_and(test_preds_weak > ms_2[i], test_preds_weak <= ms_2[i + 1])] = i
    for i in range(2):
        test_truth_a2_weak[np.logical_and(test_truth_weak > ms_2[i], test_truth_weak <= ms_2[i + 1])] = i

    # two classes{[-1.0, 0.0], (0.0, 1.0]}
    ms_2 = [-1.01, 0.0, 1.01]
    test_preds_a2 = test_preds.copy()
    test_truth_a2 = test_truth.copy()
    for i in range(2):
        test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
    for i in range(2):
        test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

    # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
    ms_3 = [-1.01, -0.1, 0.1, 1.01]
    test_preds_a3 = test_preds.copy()
    test_truth_a3 = test_truth.copy()
    for i in range(3):
        test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
    for i in range(3):
        test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

    # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
    ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
    test_preds_a5 = test_preds.copy()
    test_truth_a5 = test_truth.copy()
    for i in range(5):
        test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
    for i in range(5):
        test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a2 = __multiclass_acc(test_preds_a2, test_truth_a2)
    mult_a2_weak = __multiclass_acc(test_preds_a2_weak, test_truth_a2_weak)
    mult_a3 = __multiclass_acc(test_preds_a3, test_truth_a3)
    mult_a5 = __multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')
    r2 = r2_score(test_truth, test_preds)

    eval_results = {
        "Mult_acc_2": mult_a2,
        "Mult_acc_2_weak": mult_a2_weak,
        "Mult_acc_3": mult_a3,
        "Mult_acc_5": mult_a5,
        "F1_score": f_score,
        "MAE": mae,
        "Corr": corr,  # Correlation Coefficient
        "R_squre": r2
    }
    return eval_results["Mult_acc_2"],eval_results["F1_score"],eval_results["Mult_acc_2_weak"],eval_results["Corr"],eval_results["R_squre"],eval_results["MAE"]