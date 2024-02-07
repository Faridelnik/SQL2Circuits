import json
import os
from main_trainer import SQL2Circuits

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    #config.update('jax_platform_name', 'cpu')
except ModuleNotFoundError:
    pass

this_folder = os.path.abspath(os.getcwd())
configurations = json.load(open("sql2circuits_config.json", "r"))
workload_type = configurations["workload_types"][0]
qc_framework = configurations["qc_frameworks"][1]
classical_optimizer = configurations["classical_optimizers"][4]
measurement = configurations["measurements"][0]
circuit_architecture = configurations["circuit_architectures"][1]
learning_rate = 0.07
database = ["IMDB", "F1Data"]

model = SQL2Circuits(database = database[1],
                     run_id = 1,
                     classification = 1,
                     circuit_architecture = circuit_architecture,
                     qc_framework = qc_framework, 
                     classical_optimizer = classical_optimizer, 
                     measurement = measurement, 
                     workload_type = workload_type, 
                     total_number_of_queries = 50,
                     initial_number_of_circuits = 30, 
                     number_of_circuits_to_add = 20,
                     iterative = True,
                     epochs = 20,
                     learning_rate=learning_rate)
print("everything is ready for training")
model.train()