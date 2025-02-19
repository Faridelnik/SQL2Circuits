# -*- coding: utf-8 -*-

import random
import warnings
import os
import numpy
from circuit_preparation.circuits import Circuits
from data_preparation.database import Database
from data_preparation.prepare import DataPreparation
from data_preparation.queries import QueryGenerator
from evaluation.evaluation import Evaluation
from training.functions.pennylane_functions import make_pennylane_pred_fn, make_pennylane_pred_fn_for_gradient_descent
from training.trainers.lambeq_optax import LambeqTrainerJAX
from training.data_preparation_manager import DataPreparationManager
from training.trainers.lambeq_noisyopt import LambeqTrainer
from training.trainers.pennylane_noisyopt import PennylaneTrainer
from training.trainers.pennylane_optax import PennylaneTrainerJAX
from training.utils import *
import pickle

warnings.filterwarnings('ignore')
this_folder = os.path.abspath(os.getcwd())
SEED = 0
rng = numpy.random.default_rng(SEED)
numpy.random.seed(SEED)


class SQL2Circuits():

    def __init__(self, 
                 run_id,
                 classification,
                 circuit_architecture,
                 seed_file, 
                 qc_framework, 
                 classical_optimizer, 
                 measurement, 
                 workload_type, 
                 initial_number_of_circuits, 
                 number_of_circuits_to_add, 
                 iterative,
                 epochs,
                 learning_rate = None,
                 identifier = None):

        print("The selected configuration is: ")
        print("Run id: ", run_id)
        print("Seed file: ", seed_file)
        print("Quantum circuit framework: ", qc_framework)
        print("Classical optimizer: ", classical_optimizer)
        print("Measurement: ", measurement)
        print("Workload type: ", workload_type)
        print("Initial number of circuits: ", initial_number_of_circuits)
        print("Number of circuits to add: ", number_of_circuits_to_add)
        print("Iterative: ", iterative)
        print("Classification: ", 2**classification)
        print("Epochs: ", epochs)
        print("Learning rate: ", learning_rate)
        print("Circuit architecture: ", circuit_architecture)

        self.run_id = run_id
        self.classification = classification
        self.seed_file = seed_file
        self.qc_framework = qc_framework
        self.classical_optimizer = classical_optimizer
        self.measurement = measurement
        self.workload_type = workload_type
        self.initial_number_of_circuits = initial_number_of_circuits
        self.number_of_circuits_to_add = number_of_circuits_to_add
        self.iterative = iterative
        if identifier is None:
            self.identifier = str(run_id) + "_" + qc_framework + "_" + classical_optimizer + "_" + measurement + "_" + circuit_architecture + "_" + workload_type + "_" + str(initial_number_of_circuits) + "_" + str(number_of_circuits_to_add) + "_" + str(learning_rate).replace(".", "") + "_" + str(2**classification)
        else:
            self.identifier = identifier
        self.result = None
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.a = 0.053
        self.c = 0.00185
        self.circuit_architecture = circuit_architecture
        
        database = Database("IMDB")
        generator = QueryGenerator(self.run_id, workload_type = self.workload_type, database = "IMDB", query_seed_file_path = self.seed_file)
        query_file = generator.get_query_file()
        self.data_preparator = DataPreparation(run_id, query_file, database = database, workload_type = self.workload_type, classification = self.classification)
        self.total_number_of_circuits = len(self.data_preparator.get_training_data_labels())

        output_folder = this_folder + "//circuit_preparation//data//circuits//" + str(run_id) + "//"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print("The new directory: ", output_folder, " is created for circuits.")

        self.results_folder = this_folder + "//training//results//" + self.identifier + "//"
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
            print("The new directory: ", self.results_folder, " is created for results.")

        info = {
            "run_id": self.run_id,
            "seed_file": self.seed_file,
            "qc_framework": self.qc_framework,
            "classical_optimizer": self.classical_optimizer,
            "measurement": self.measurement,
            "workload_type": self.workload_type,
            "initial_number_of_circuits": self.initial_number_of_circuits,
            "number_of_circuits_to_add": self.number_of_circuits_to_add,
            "iterative": self.iterative,
            "classification": 2**self.classification,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "classes": ' '.join(str(x) for x in self.data_preparator.get_classes()),
            "circuit_architecture": self.circuit_architecture
        }
        json.dump(info, open("training_stats.json", "w"), indent = 4)
        self.query_file = query_file 
        self.output_folder = output_folder

        self.circuits = Circuits(self.run_id, 
                                 self.query_file, 
                                 self.output_folder, 
                                 self.classification, 
                                 self.measurement,
                                 self.circuit_architecture,
                                 write_cfg_to_file = True, 
                                 write_pregroup_to_file=True,
                                 generate_cfg_png_diagrams = True,
                                 generate_pregroup_png_diagrams = True,
                                 generate_circuit_png_diagrams = True)
        self.circuits.execute_full_transformation()


    def train(self):
        if self.iterative and self.classical_optimizer == "noisyopt":
            self.iterative_train_noisyopt()
        elif not self.iterative and self.classical_optimizer == "noisyopt":
            self.single_train_noisyopt()
        elif self.iterative and "optax" in self.classical_optimizer:
            self.iterative_train_optax()
        

    def iterative_train_noisyopt(self, hyperparameter_file = None):
        for i in range(self.initial_number_of_circuits, 
                       self.total_number_of_circuits + self.number_of_circuits_to_add, 
                       self.number_of_circuits_to_add):
            if i > self.total_number_of_circuits:
                i = self.total_number_of_circuits
            hyperparameter_file = "training//hyperparameter_results//" + str(self.identifier) + "//" + str(i) + "_" + str(self.run_id) + "_cv_results_.json"
            self.train_noisyopt(i, hyperparameter_file)


    def single_train_noisyopt(self, hyperparameter_file = None):
        self.train_noisyopt(self.total_number_of_circuits, hyperparameter_file)


    def train_noisyopt(self, number_of_selected_circuits, hyperparameter_file = None):
        if hyperparameter_file is not None:
            # "training//results//" + str(run_id) + "//" + str(i) + "_" + str(run_id) + "_cv_results_.json"
            if os.path.exists(hyperparameter_file):
                with open(hyperparameter_file, "r") as f:
                    param_file = json.load(f)
                    self.a = param_file["best_params"]["a"]
                    self.c = param_file["best_params"]["c"]
            else:
                print("The hyperparameter file does not exist. The default values are used.")

        sf = DataPreparationManager(self.run_id, self.data_preparator, self.circuits, number_of_selected_circuits, self.qc_framework)
        X_train = sf.get_X_train()
        y = sf.get_training_labels()
        validation_circuits = sf.get_X_valid()
        validation_labels = sf.get_validation_labels()
        test_circuits = sf.get_X_test()
        test_labels = sf.get_test_labels()
        params = sf.get_lambeq_symbols()

        if self.qc_framework == "lambeq":
            trainer = LambeqTrainer(self.identifier,
                                circuits = self.circuits,
                                workload_type = self.workload_type,
                                classification = self.classification,
                                a = self.a,
                                c = self.c,
                                epochs = self.epochs,
                                classical_optimizer = self.classical_optimizer,
                                measurement = self.measurement)
            self.result = trainer.fit_with_lambeq_noisyopt(X_train, 
                                                           y, 
                                                           validation_circuits = validation_circuits, 
                                                           validation_labels = validation_labels, 
                                                           save_parameters = True)
            evaluator = Evaluation(self.run_id, self.identifier, self.result, test_circuits, test_labels, params)
            evaluator.evaluate_lambeq_on_test_set(number_of_selected_circuits)

        elif self.qc_framework == "pennylane":
            trainer = PennylaneTrainer(self.identifier,
                                circuits = self.circuits,
                                workload_type = self.workload_type,
                                classification = self.classification,
                                a = self.a,
                                c = self.c,
                                epochs = self.epochs,
                                classical_optimizer = self.classical_optimizer,
                                measurement = self.measurement)
            qml_params = sf.get_qml_train_symbols()
            self.result = trainer.fit_with_pennylane_noisyopt(X_train, 
                                                              y, 
                                                              validation_circuits = validation_circuits,
                                                              validation_labels = validation_labels,
                                                              qml_params = qml_params, 
                                                              save_parameters = True)
            evaluator = Evaluation(self.run_id, self.identifier, self.result, test_circuits, test_labels)
            evaluator.evaluate_pennylane_on_test_set(number_of_selected_circuits)
        return self.result


    def train_optax(self, number_of_circuits):
        sf = DataPreparationManager(self.run_id, self.data_preparator, self.circuits, number_of_circuits, self.qc_framework)
        X_train = sf.get_X_train()
        y = sf.get_training_labels()
        validation_circuits = sf.get_X_valid()
        validation_labels = sf.get_validation_labels()
        test_circuits = sf.get_X_test()
        test_labels = sf.get_test_labels()
        params = sf.get_lambeq_symbols()
        
        if self.qc_framework == "pennylane":
            params = sf.get_qml_train_symbols()
            trainer = PennylaneTrainerJAX(self.identifier, 
                                          self.classical_optimizer, 
                                          params, 
                                          self.learning_rate, 
                                          self.epochs, 
                                          self.classification)
            self.result = trainer.train(X_train, y, validation_circuits = validation_circuits, validation_labels = validation_labels)
            evaluator = Evaluation(self.run_id, self.identifier, self.result, test_circuits, test_labels)
            evaluator.evaluate_pennylane_optax_on_test_set(number_of_circuits)
        elif self.qc_framework == "lambeq":
            params = sf.get_lambeq_symbols()
            trainer = LambeqTrainerJAX(self.identifier, 
                                       self.classical_optimizer, 
                                       params, 
                                       self.learning_rate, 
                                       self.epochs, 
                                       self.classification)
            self.result = trainer.train(X_train, y, validation_circuits = validation_circuits, validation_labels = validation_labels)
            evaluator = Evaluation(self.run_id, self.identifier, self.result, test_circuits, test_labels, params)
            evaluator.evaluate_lambeq_on_test_set(number_of_circuits)
        with open(self.results_folder + str(number_of_circuits) + "_optax_results_.pkl", "wb") as f:
            pickle.dump(self.result, f)
        
        return self.result


    def iterative_train_optax(self):
        for i in range(self.initial_number_of_circuits, 
                       self.total_number_of_circuits + self.number_of_circuits_to_add, 
                       self.number_of_circuits_to_add):
            if i > self.total_number_of_circuits:
                i = self.total_number_of_circuits
            self.train_optax(i)
    
            
    def evaluate_on_IQM(self):    
        self.circuits = Circuits(self.run_id, 
                                 self.query_file, 
                                 self.output_folder, 
                                 self.classification, 
                                 "state",
                                 self.circuit_architecture)
        self.circuits.execute_full_transformation()
        #self.circuits.generate_pennylane_circuits()
        
        sf = DataPreparationManager(self.run_id, 
                                    self.data_preparator, 
                                    self.circuits, 
                                    "all", 
                                    self.qc_framework)
        train_circuits = sf.get_X_train()
        train_labels = sf.get_training_labels()
        
        valid_circuits = sf.get_X_valid()
        valid_labels = sf.get_validation_labels()
        
        test_circuits = sf.get_X_test()
        test_labels = sf.get_test_labels()
        
        params = sf.get_qml_train_symbols()
        print("Number of parameters: ", len(params))
        
        #train_pred_fn = make_pennylane_pred_fn(train_circuits, params, self.classification)
        #valid_pred_fn = make_pennylane_pred_fn(valid_circuits, params, self.classification)
        #test_pred_fn = make_pennylane_pred_fn(test_circuits, params, self.classification)
        
        train_pred_fn = make_pennylane_pred_fn_for_gradient_descent(train_circuits)
        valid_pred_fn = make_pennylane_pred_fn_for_gradient_descent(valid_circuits)
        test_pred_fn = make_pennylane_pred_fn_for_gradient_descent(test_circuits)
        
        result_file = self.results_folder + "optimized_params.json"
        optimized_params = None
        with open(result_file, "rb") as f:
            optimized_params = json.load(f)
        print("Number of optimized parameters: ", len(optimized_params))
        for case in ["train", "valid", "test"]:
            if case == "train":
                pred_fn = train_pred_fn
                labels = train_labels
            elif case == "valid":
                pred_fn = valid_pred_fn
                labels = valid_labels
            else:
                pred_fn = test_pred_fn
                labels = test_labels
            pred = pred_fn(optimized_params)
            print("Predictions: ", pred)
            acc = multi_class_acc(pred, labels)
            test_result_file = this_folder + "//training//results//IQM//accuracy_iqm.json"
            # Append and save accuracy to the file with the identifier
            if os.path.exists(test_result_file):
                with open(test_result_file, "r") as f:
                    result = json.load(f)
                    if self.identifier in result:
                        result[self.identifier][case] = acc
                    else:
                        result[self.identifier] = {}
                        result[self.identifier][case] = acc
            else:
                result = {}
                result[self.identifier] = {}
                result[self.identifier][case] = acc
            
            with open(test_result_file, "w") as f:
                json.dump(result, f, indent = 4)
        
        
        
        
        