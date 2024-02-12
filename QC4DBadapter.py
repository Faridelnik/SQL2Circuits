import pickle
try:
    from antlr4 import *
except ModuleNotFoundError:
    print("Please install antlr4-python3-runtime to use the parser.")
import json
import os
from discopy.grammar.pregroup import Ty, Functor
from discopy.utils import dumps, loads
from circuit_preparation.diagrams.pregroupFunctorMappings import count_boxes, object_mapping, arrow_mapping
from circuit_preparation.diagrams.cupRemoveFunctorMappings import cup_remove_arrow_mapping, cup_remove_arrow_mapping2
from circuit_preparation.diagrams.parser.SQLiteLexer import SQLiteLexer
from circuit_preparation.diagrams.parser.SQLiteParser import SQLiteParser
from circuit_preparation.diagrams.parser.SQLiteParserListener import SQLiteParserListener
from lambeq.ansatz import IQPAnsatz, Sim14Ansatz, Sim15Ansatz, StronglyEntanglingAnsatz
from training.functions.pennylane_functions import predict_circuit
from training.pennylane_circuit import PennylaneCircuit


#what should be in the results:
# - optimised parameters
# - set of free symbols
# - centers of clusters


if 'num_classes' in locals():
    print("Number of classes:", num_classes)
else:
    print("Number of classes not provided.")

with open(file_with_optimized_params, 'rb') as f:
    params = pickle.load(f)

cup_removal_functor = Functor(ob = lambda x: x, ar = lambda f: cup_remove_arrow_mapping(f))
cup_removal_functor2 = Functor(ob = lambda x: x, ar = lambda f: cup_remove_arrow_mapping2(f))

n, s = Ty('n'), Ty('s')

def perform_full_transformation(query):

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
    classification = 4
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
    full_symbol_to_index.update(symbol_to_index)

    # Produces a dictionary like {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0} 
    # where the wires 0 and 1 are the classifying wires
    post_selection = dict([(i, 0) for i in range(classification, n_qubits)])
    valid_states = np.array(get_valid_states(n_qubits, post_selection))

    interface = "best"
    diff_method = "best"

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

    predicted_class = predict_circuit(circuit, params, n_qubits, classification)










