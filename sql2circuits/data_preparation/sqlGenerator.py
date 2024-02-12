import json
import os
from itertools import combinations
import random
import math
random.seed(0)

class SQLGenerator:

  """
    A class that generates SQL queries by combining filters, joins, and selects from a query seed file. 
    The generated queries are used for training, testing, and validation of SQL2Circuits model.
    """

  def __init__(self, id, workload_type, database, total_number_of_queries = 50, test_query_ratio = 0.3, validation_query_ratio = 0.3) -> None:
      self.id = id
      self.database = database
      self.workload_type = workload_type
      self.total_number_of_queries = total_number_of_queries
      self.test_query_ratio = test_query_ratio
      self.validation_query_ratio = validation_query_ratio
      self.this_folder = os.path.abspath(os.getcwd())
      self.queries = None
      self.path_for_queries = self.this_folder + "/data_preparation/queries/" + self.workload_type + "/"
      self.query_file = self.path_for_queries + str(self.id) + ".json"
      
      if not os.path.exists(self.path_for_queries):
          os.makedirs(self.path_for_queries)
          print("The new directory: ", self.path_for_queries, " is created for queries.")
      if os.path.isfile(self.query_file):
          print("Query file already exists.")
          self.queries = json.load(open(self.query_file, "r"))
      else:
          self.queries = self.construct_queries(max_num_of_joins = 4)
          with open(self.query_file, "w") as output:
              json.dump(self.queries, output, indent = 4)

    
  def get_query_file(self):
      return self.query_file
  
  def construct_queries(self, max_num_of_joins):
    """
    Constructs SQL queries from the given queries and generates training, test and validation queries based on the given ratios.
    
    Args:
    - queries: list of dictionaries containing filters, joins, selects and table aliases
    
    Returns:
    - A dictionary containing the statistics of the generated queries and the generated queries themselves.
    """
    result_queries = []
    tables_with_alises = self.database.get_tables_names()
    all_columns = self.database.get_all_columns_from_all_tables()
    columns_for_possible_joins = self.database.get_tables_with_same_columns()

    #dictionary_of_allowed_selections = {1: 20, 2: 20, 3: 15, 4: 10}
    dictionary_of_allowed_selections = {1: 5, 2: 5, 3: 3, 4: 1}

    for i in range(math.ceil(self.total_number_of_queries/max_num_of_joins)):
        
        for j in range(max_num_of_joins):
            
            query = "SELECT "

            num_joins_to_select = random.randint(1, j+1)
            selected_columns = random.sample(columns_for_possible_joins.keys(), num_joins_to_select)

            all_involved_tables = []
            where_statement = ""

            for column in selected_columns:
                
                involved_tables = columns_for_possible_joins[column]
                result_list = involved_tables[1:-1].split(',')

                if len(result_list) == 2:
                    table1 = result_list[0]
                    table2 = result_list[1]
                else:
                    two_random_tables = random.sample(result_list, 2)
                    table1 = two_random_tables[0]
                    table2 = two_random_tables[1]

                where_statement += tables_with_alises[table1] + "." + column + " = " + tables_with_alises[table2] + "." + column + " AND "
                all_involved_tables.append(table1)
                all_involved_tables.append(table2)

            num_of_columns_to_select = random.randint(1, dictionary_of_allowed_selections[j+1])

            list_of_all_involved_tables = list(set(all_involved_tables))

            counter = 0
            selected_tables = ""
            for table in list_of_all_involved_tables:
                columns = all_columns[table]
                selected_tables += table + " AS " + tables_with_alises[table] + ", "
                for c in columns:
                    counter = counter + 1
                    if counter > num_of_columns_to_select:
                        break
                    query+= tables_with_alises[table] + "." + c + ", "

            query = query[:-2] + " FROM " + selected_tables
            query = query[:-2] + " WHERE " + where_statement
            query = query[:-5]
            query += ";"
            result_queries.append(query)
        
    #return result_queries

    probabilities = [1.0 - self.test_query_ratio - self.validation_query_ratio, 
                      self.test_query_ratio, 
                      self.validation_query_ratio]

    result = random.choices(population=[1, 2, 3], weights=probabilities, k=len(result_queries))

    query_range = range(len(result_queries))

    training_queries = [ { "id": i, "query": result_queries[i] } for i in query_range if result[i] == 1]
    test_queries = [{ "id": i, "query": result_queries[i] } for i in query_range if result[i] == 2]
    validation_queries = [{ "id": i, "query": result_queries[i] } for i in query_range if result[i] == 3]


    return {"stats": {"total_number_of_queries": len(result_queries),
                    "number_of_training_queries": len(training_queries), 
                    "number_of_test_queries": len(test_queries), 
                    "number_of_validation_queries": len(validation_queries)},
            "queries": { 
                    "training": training_queries, 
                    "test": test_queries, 
                    "validation": validation_queries
                    }}
