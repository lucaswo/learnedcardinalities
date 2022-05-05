import numpy as np


# Helper functions for data processing

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names


def get_all_table_names(tables):
    table_names = set()
    for query in tables:
        for table in query:
            table_names.add(table)
    return table_names


def get_all_operators(predicates):
    operators = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                operator = predicate[1]
                operators.add(operator)
    return operators


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot


def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def get_min_max_vals(predicates, column_names):
    min_max_vals = {t: [float('inf'), float('-inf')] for t in column_names}
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                val = float(predicate[2])
                if val < min_max_vals[column_name][0]:
                    min_max_vals[column_name][0] = val
                if val > min_max_vals[column_name][1]:
                    min_max_vals[column_name][1] = val
    return min_max_vals


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)


def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)


def encode_samples(tables, samples, table2vec):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector
            if len(samples) > 0:
                sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc


# def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
#     predicates_enc = []
#     joins_enc = []
#     for i, query in enumerate(predicates):
#         predicates_enc.append(list())
#         joins_enc.append(list())
#         for predicate in query:
#             if len(predicate) == 3:
#                 # Proper predicate
#                 column = predicate[0]
#                 operator = predicate[1]
#                 val = predicate[2]
#                 norm_val = normalize_data(val, column, column_min_max_vals)

#                 pred_vec = []
#                 pred_vec.append(column2vec[column])
#                 pred_vec.append(op2vec[operator])
#                 pred_vec.append(norm_val)
#                 pred_vec = np.hstack(pred_vec)
#             else:
#                 pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1))

#             predicates_enc[i].append(pred_vec)

#         for predicate in joins[i]:
#             # Join instruction
#             join_vec = join2vec[predicate]
#             joins_enc[i].append(join_vec)
#     return predicates_enc, joins_enc

def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, num_buckets):
    predicates_enc = []
    joins_enc = []

    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        joins_enc.append(list())
        
        reduced_min_max = {k:v for k,v in column_min_max_vals.items() if k in column2vec.keys()}
        pred_vec = vectorize_attribute_domains_no_disjunctions(query, reduced_min_max, num_buckets, column2vec)
        #pred_vec = vectorize_query_range(query, column_min_max_vals, column2vec, op2vec)
        predicates_enc[i] = pred_vec

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return predicates_enc, joins_enc


def vectorize_query_range(predicates, min_max, column2vec, op2vec):
    #total_columns = len(min_max)
    totalfeaturevec = list()
    
    #collect bounds
    bounds = dict()
    for exp in predicates:
        if len(exp) == 3: # proper predicate
            exp[-1] = min(max(min_max[exp[0]][0], float(exp[-1])), min_max[exp[0]][1])
            if exp[0] not in bounds.keys():
                bounds[exp[0]] = list()
            bounds[exp[0]].append(exp[1:])
        else:
            return [np.zeros(len(column2vec) + 2*len(op2vec)+2)]
    
    for pred, limits in bounds.items():
        # extend incomplete bounds and single bounds <>, =
        if len(limits) < 2:
            if limits[0][0] == "<>" or limits[0][0] == "=":
                limits.append(limits[0])
            elif ">" in limits[0][0]:
                limits.append(["<", min_max[pred][1]])
            elif "<" in limits[0][0]:
                limits.insert(0, [">", min_max[pred][0]])
        
        vector = np.zeros(2*len(op2vec)+2)
        offset = 0
        # only upper and lower -> offset = 0 then offset = 4
        # limits[:2] contains lower and upper bound (limit[2:] contains <> constraints)
        #idx = list(sorted(min_max.keys())).index(pred)
        for op, bound in limits[:2]:
            vector[offset:offset+3] = op2vec[op]
            if bound is None:
                vector[offset+3] = 0
            else:
                vector[offset+3] = normalize_data(bound, pred, min_max)
            
            offset += 4

        totalfeaturevec.append(np.concatenate((column2vec[pred], vector)))
    
    return totalfeaturevec


def vectorize_attribute_domains_no_disjunctions(predicates, min_max, max_bucket_count, column2vec):
    _, atomar_buckets, bounds, not_values = prepare_data_structures(min_max, max_bucket_count)
    feature_vectors = dict()

    for exp in predicates:
        if len(exp) == 3:  
            attr, op, val = exp
            if attr not in feature_vectors.keys():
                feature_vectors[attr] = np.ones(int(max_bucket_count) + 1)

            val = min(max(min_max[attr][0], float(val)), min_max[attr][1])
            attr_feature_vec = feature_vectors[attr]
            domainrange = min_max[attr][1] - min_max[attr][0] + 1
            positionval = val - min_max[attr][0]
            # k = positionval / domainrange in [0,1), floor(k * len(vector)) gives number [0, len(vector)-1]
            val_bucket_idx = int(float(positionval) / domainrange * len(feature_vectors[attr]))
            add_simplepred_to_featurevec(attr_feature_vec, val_bucket_idx, attr, op, val, 
                                        min_max, atomar_buckets, bounds, not_values)

    for attr in feature_vectors.keys():
        # set covered domain ratio
        domainrange = min_max[attr][1] - min_max[attr][0]
        queryrange = bounds[attr][1] - bounds[attr][0]
        notsum = sum( [1 for x in not_values[attr] if bounds[attr][0] <= x <= bounds[attr][1]])
        queryrange = max(queryrange - notsum,  0)
        feature_vectors[attr][-1] = queryrange / domainrange
        feature_vectors[attr] = np.concatenate((column2vec[attr], feature_vectors[attr])) # add column reference to identify pred

    # query without any predicates
    if len(feature_vectors) == 0:
        totalfeaturevec = [np.zeros(len(column2vec) + max_bucket_count + 1)]
    else:
        totalfeaturevec = list(feature_vectors.values())
    #print(f"{totalfeaturevec=}")
    return totalfeaturevec

#helper function
def add_simplepred_to_featurevec(attr_feature_vec, val_bucket_idx, attr, op, val, min_max, atomar_buckets, bounds, not_values):
    if op == "=" or op == "IS":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[0 : val_bucket_idx] = 0
        attr_feature_vec[val_bucket_idx+1 : -1] = 0
        bounds[attr][0] = val
        bounds[attr][1] = val+1
    elif op == ">":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 0 if atomar_buckets[attr] else 0.5
        attr_feature_vec[0 : val_bucket_idx] = 0
        bounds[attr][0] = max(bounds[attr][0], min(val+1, min_max[attr][1]))
    elif op == "<":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[val_bucket_idx+1 : -1] = 0
        bounds[attr][1] = min(bounds[attr][1], max(val-1, min_max[attr][0]))
    elif op == "<=":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[val_bucket_idx+1 : -1] = 0
        bounds[attr][1] = min(bounds[attr][1], val)
    elif op == ">=":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 1 if atomar_buckets[attr] else 0.5
        attr_feature_vec[0 : val_bucket_idx] = 0
        bounds[attr][0] = max(bounds[attr][0], val)
    elif op == "<>" or op == "!=":
        if attr_feature_vec[val_bucket_idx] == 1:
            attr_feature_vec[val_bucket_idx] = 0 if atomar_buckets[attr] else 0.5
        not_values[attr].append(val)
    else:
        raise SystemExit("Unknown operator", op)
    
    return attr_feature_vec

# helper function
def prepare_data_structures(min_max, max_bucket_count):
    feature_vectors = dict() # dict of floats by attribute
    atomar_buckets = dict() # dict of booleans by attribute
    not_values = dict() # dict of list by attribute

    for attr, domain in min_max.items():
        domainrange = domain[1] - domain[0]
        if max_bucket_count < domainrange:
            atomar_buckets[attr] = False
        else:
            atomar_buckets[attr] = True
        feature_vectors[attr] = np.ones(int(max_bucket_count) + 1) # last one is for covered ratio
        bounds = {attr : list(vals) for attr, vals in min_max.items()}
        not_values[attr] = []

    return feature_vectors, atomar_buckets, bounds, not_values
