import pickle
from antlr4 import *
import json
import os
from discopy.grammar.pregroup import Ty, Functor
# from discopy.utils import dumps, loads
from circuit_preparation.diagrams.pregroupFunctorMappings import count_boxes, object_mapping, arrow_mapping
from circuit_preparation.diagrams.cupRemoveFunctorMappings import cup_remove_arrow_mapping, cup_remove_arrow_mapping2
from circuit_preparation.diagrams.parser.SQLiteLexer import SQLiteLexer
from circuit_preparation.diagrams.parser.SQLiteParser import SQLiteParser
from circuit_preparation.diagrams.parser.SQLiteParserListener import SQLiteParserListener
from lambeq.ansatz import IQPAnsatz, Sim14Ansatz, Sim15Ansatz, StronglyEntanglingAnsatz
from training.functions.pennylane_functions import get_valid_states
from training.pennylane_circuit import PennylaneCircuit
from discopy.quantum.pennylane import to_pennylane
import numpy as np
from sympy.core.symbol import Symbol
from sympy import default_sort_key
import sys

#what should be in the results:
# - optimised parameters
# - centers of clusters

# what should be passed:
# - number of classes
# - name of the database

cup_removal_functor = Functor(ob = lambda x: x, ar = lambda f: cup_remove_arrow_mapping(f))
cup_removal_functor2 = Functor(ob = lambda x: x, ar = lambda f: cup_remove_arrow_mapping2(f))

n, s = Ty('n'), Ty('s')

def perform_full_transformation(query, num_classes, optimized_params):

    classification = int(np.log2(num_classes))

    # create CFG diagram
    input_stream = InputStream(query)
    lexer = SQLiteLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = SQLiteParser(stream)
    tree = parser.parse()
    walker = ParseTreeWalker()
    listener = SQLiteParserListener(parser)
    walker.walk(listener, tree)
    diagram = listener.get_final_diagram()

    # create pregroup grammar diagram
    num_of_result_columns = count_boxes(diagram, "result-column")
    num_of_result_columns += count_boxes(diagram, "result-column-with-alias")
    num_of_tables = count_boxes(diagram, "table")
    num_of_tables += count_boxes(diagram, "table-with-alias")
    
    Rewriter = Functor(ob = lambda x: object_mapping(x, num_of_result_columns, num_of_tables), 
                        ar = lambda f: arrow_mapping(f, num_of_result_columns, num_of_tables))
    
    pregroup_diagram = Rewriter(diagram)

    # remove cups and simplify
    cupless_pregroup_diagram = cup_removal_functor(pregroup_diagram.normal_form()).normal_form() # type: ignore
    cupless_pregroup_diagram = cup_removal_functor2(cupless_pregroup_diagram).normal_form() # type: ignore
    # Flip the diagram, otherwise the order of the gates is reversed
    cupless_pregroup_diagram = cupless_pregroup_diagram[::-1]

    #create circuit ansatz
    circuit_architecture = "Sim14Ansatz"
    n_wire_count = 1
    layers = 1
    single_qubit_params = 3

    if circuit_architecture == "IQPAnsatz":
        ansatz = IQPAnsatz({n: n_wire_count, 
                                s: classification}, 
                                n_layers = layers, 
                                n_single_qubit_params = single_qubit_params)
    elif circuit_architecture == "Sim14Ansatz":
        ansatz = Sim14Ansatz({n: n_wire_count, 
                                s: classification}, 
                                n_layers = layers, 
                                n_single_qubit_params = single_qubit_params)
    elif circuit_architecture == "Sim15Ansatz":
        ansatz = Sim15Ansatz({n: n_wire_count, 
                                s: classification}, 
                                n_layers = layers, 
                                n_single_qubit_params = single_qubit_params)
    elif circuit_architecture == "StronglyEntanglingAnsatz":
        ansatz = StronglyEntanglingAnsatz({n: n_wire_count, 
                                s: classification}, 
                                n_layers = layers, 
                                n_single_qubit_params = single_qubit_params)

    circuit_diagram = ansatz(cupless_pregroup_diagram)

    # generate pennylane circuit
    symbols = set([Symbol(str(elem)) for c in circuit_diagram for elem in c.free_symbols])
    symbols = list(sorted(symbols, key = default_sort_key))
    
    pennylane_circuit = to_pennylane(circuit_diagram)
    ops = pennylane_circuit._ops
    params = pennylane_circuit._params
    pennylane_wires = pennylane_circuit._wires
    n_qubits = pennylane_circuit._n_qubits
    param_symbols = [[sym[0].as_ordered_factors()[1]] if len(sym) > 0 else [] for sym in params]
    #print("Param symbols: ", param_symbols)
    symbol_to_index = {}

    for sym in param_symbols:
        if len(sym) > 0:
            symbol_to_index[sym[0]] = symbols.index(sym[0])
   
    # Produces a dictionary like {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0} 
    # where the wires 0 and 1 are the classifying wires
    post_selection = dict([(i, 0) for i in range(classification, n_qubits)])
    valid_states = np.array(get_valid_states(n_qubits, post_selection))

    interface = "auto"
    diff_method = "best"
    measurement = "state"

    qml_circuit = PennylaneCircuit(ops, 
                                   params, 
                                   pennylane_wires, 
                                   n_qubits, 
                                   param_symbols, 
                                   symbol_to_index, 
                                   symbols, 
                                   valid_states,
                                   measurement,
                                   interface, 
                                   diff_method)
    
    prediction = qml_circuit.eval_qml_circuit_with_post_selection(optimized_params)
    return prediction

def find_directories(root_dir, *keywords):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if all(keyword in dirname for keyword in keywords):
                matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs

def find_pkl_files(root_dir):
    pkl_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pkl"):
                pkl_files.append(os.path.join(dirpath, filename))
    return pkl_files

def predict_execution_time(query):

    num_classes = 4

    #script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = "/qc4db/SQL2Circuits/sql2circuits/training/results"
    keywords = ["F1Data", str(num_classes)+"_classes"]
    
    matching_directories = find_directories(root_directory, *keywords)
    # if matching_directories:
    #     print("Matching directories:")
    #     for directory in matching_directories:
    #         print(directory)
    # else:
    #     print("No directories found matching the criteria.")

    directory_with_results = matching_directories[0]
    pkl_files = find_pkl_files(directory_with_results)

    if pkl_files:
        pkl_files.sort()  # Sorting files alphabetically
        last_pkl_file = pkl_files[-1]  # Selecting the last file
        #print("Last .pkl file found:", last_pkl_file)
    else:
        print("No .pkl files found in the directory.")

    file_with_optimized_params = last_pkl_file
    with open(file_with_optimized_params, 'rb') as f:
        optimized_params = pickle.load(f)

    #query = "SELECT b2.constructorresultsid FROM constructorresults AS b2, drivers AS f6, circuits AS a1, status AS n14, results AS k11, sprintresults AS m13 WHERE b2.status = n14.status AND a1.url = f6.url AND m13.fastestlaptime = k11.fastestlaptime;"
        
    #print("query received by python: ", query)

    prediction = perform_full_transformation(query, num_classes, optimized_params)
    #print(prediction)

    with open(directory_with_results + '/training_stats.json', 'r') as f:
        data = json.load(f)

    # Extract the cluster centers
    cluster_centers_str = data.get('cluster centers', '')
    cluster_centers = [float(center.strip('[]')) for center in cluster_centers_str.split()]

    #print("Cluster Centers:", cluster_centers)

    max_index = np.argmax(prediction)
    estimated_runtime = cluster_centers[max_index]

    #print("Estimated runtime: ", estimated_runtime)

    return estimated_runtime



if __name__ == "__main__":

    # local_vars = locals()
    # print("locals ", local_vars)

    # if 'num_classes' in locals():
    #     print("Number of classes:", num_classes)
    # else:
    #     print("Number of classes not provided.")

    num_classes = 4

    #script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = "/qc4db/SQL2Circuits/sql2circuits/training/results"
    keywords = ["F1Data", str(num_classes)+"_classes"]
    
    matching_directories = find_directories(root_directory, *keywords)
    # if matching_directories:
    #     print("Matching directories:")
    #     for directory in matching_directories:
    #         print(directory)
    # else:
    #     print("No directories found matching the criteria.")

    directory_with_results = matching_directories[0]
    pkl_files = find_pkl_files(directory_with_results)

    if pkl_files:
        pkl_files.sort()  # Sorting files alphabetically
        last_pkl_file = pkl_files[-1]  # Selecting the last file
        #print("Last .pkl file found:", last_pkl_file)
    else:
        print("No .pkl files found in the directory.")

    file_with_optimized_params = last_pkl_file
    with open(file_with_optimized_params, 'rb') as f:
        optimized_params = pickle.load(f)

    #query = "SELECT b2.constructorresultsid FROM constructorresults AS b2, drivers AS f6, circuits AS a1, status AS n14, results AS k11, sprintresults AS m13 WHERE b2.status = n14.status AND a1.url = f6.url AND m13.fastestlaptime = k11.fastestlaptime;"
   
    print ("This is the name of the script: ", sys.argv[0])
    print ("Number of arguments: ", len(sys.argv))
    print ("The arguments are: " , str(sys.argv))
    query = str(sys.argv)

    prediction = perform_full_transformation(query, num_classes, optimized_params)
    #print(prediction)

    with open(directory_with_results + '/training_stats.json', 'r') as f:
        data = json.load(f)

    # Extract the cluster centers
    cluster_centers_str = data.get('cluster centers', '')
    cluster_centers = [float(center.strip('[]')) for center in cluster_centers_str.split()]

    #print("Cluster Centers:", cluster_centers)

    max_index = np.argmax(prediction)
    estimated_runtime = cluster_centers[max_index]

    print("Estimated runtime: ", estimated_runtime)

    












    










