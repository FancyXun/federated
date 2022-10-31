import tensorflow as tf
import tensorflow_federated as tff


@tff.federated_computation(tff.type_at_clients(tf.int32))
def fed_sum(x):
    return tff.federated_sum(x)


result = fed_sum([])

print(result)
assert (result == 0)
