# -*- coding: utf-8 -*-

import json
import os
from training.cost_accuracy import CostAccuracy
from training.functions.lambeq_functions import make_lambeq_cost_fn, make_lambeq_pred_fn
from training.functions.pennylane_functions import make_pennylane_cost_fn, make_pennylane_pred_fn, make_pennylane_pred_fn_for_gradient_descent
from training.utils import multi_class_acc, multi_class_loss, store_and_log
this_folder = os.path.abspath(os.getcwd())
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


class Evaluation:

    def __init__(self, run_id, identifier, classification, result_params, test_circuits, test_labels, test_params = None) -> None:
        self.run_id = run_id
        self.identifier = identifier
        self.classification = 2**classification
        self.result_params = result_params
        self.test_circuits = test_circuits
        self.test_labels = test_labels
        self.params = test_params
        self.test_result_file = "training//results//" + str(self.run_id) + "//" + str(self.run_id) + "_test_results.json"
        self.loss_function = multi_class_loss
        self.accuracy = multi_class_acc

    
    def evaluate_lambeq_on_test_set(self, iteration):
        
        test_pred_fn = make_lambeq_pred_fn(self.test_circuits, self.params)
        costs_accuracies = CostAccuracy()
        test_cost_fn = make_lambeq_cost_fn(test_pred_fn, self.test_labels, self.loss_function, self.accuracy, costs_accuracies, "test")
        test_cost_fn(self.result_params) # type: ignore
        test_accs = costs_accuracies.get_test_accs()
        test_result_file = this_folder + "//training//results//" + self.identifier + "//test_accuracy.json"
        if not os.path.isfile(test_result_file):
            with open(test_result_file, "w") as f:
                json.dump({ "results": [] }, f, indent=4)
        with open(test_result_file, "r") as f:
            file = json.load(f)
            file["results"].append({ "step": iteration, "test_accuracy": test_accs, "number_of_test_circuits": len(self.test_circuits) })
            json.dump(file, open(test_result_file, "w"), indent=4)


    def evaluate_pennylane_on_test_set(self, iteration):
        test_pred_fn = make_pennylane_pred_fn(self.test_circuits)
        costs_accuracies = CostAccuracy()
        test_cost_fn = make_pennylane_cost_fn(test_pred_fn, self.test_labels, self.loss_function, self.accuracy, costs_accuracies, "test")
        test_cost_fn(self.result_params.x) # type: ignore
        test_accs = costs_accuracies.get_test_accs()
        store_and_log(iteration, { "test_accuracy": test_accs[0] }, self.test_result_file)


    def evaluate_pennylane_optax_on_test_set(self, iteration):
        test_pred_fn = make_pennylane_pred_fn_for_gradient_descent(self.test_circuits)
        test_acc = self.accuracy(test_pred_fn(self.result_params), self.test_labels)
        test_result_file = this_folder + "/training/results/" + self.identifier + "/test_accuracy.json"
        if not os.path.isfile(test_result_file):
            with open(test_result_file, "w") as f:
                json.dump({ "results": [] }, f, indent=4)
        with open(test_result_file, "r") as f:
            file = json.load(f)
            file["results"].append({ "step": iteration, "test_accuracy": test_acc, "number_of_test_circuits": len(self.test_circuits) })
            json.dump(file, open(test_result_file, "w"), indent=4)

        test_data_predictions = test_pred_fn(self.result_params)

        actual_test_labels = np.zeros(len(test_data_predictions))
        predicted_test_labels = np.zeros(len(test_data_predictions))
        
        j=0
        for data in self.test_labels:
            for i in range(self.classification):
                if data[i]==1:
                    actual_test_labels[j] = i
            j = j + 1

        for i in range(len(test_data_predictions)):
            max_index = np.argmax(test_data_predictions[i])
            predicted_test_labels[i] = max_index

        # Calculate the confusion matrix (normalized)
        cm = confusion_matrix(actual_test_labels, predicted_test_labels, normalize='true', labels=range(self.classification))

        # Create a colorful heatmap plot
        fig = plt.figure(figsize=(8, 6))
        label_list = ['Label {}'.format(i) for i in range(self.classification)]
        sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=label_list, yticklabels=label_list)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix SQL2Circuits Optax')
        fig_name = 'confusion_matrix_' + str(self.classification) + '_classes' + str(iteration) + '_circuits.png'
        fig_path = this_folder + "/training/results/" + self.identifier + "/"
        fig.savefig(fig_path + fig_name, dpi=fig.dpi)