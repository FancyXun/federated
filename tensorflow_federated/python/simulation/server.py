# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A generic worker binary for deployment, e.g., on GCP."""
from absl import app


from tensorflow_federated.python.core.impl.executor_stacks import python_executor_stacks
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.simulation import server_utils


def main(argv):

    '''
    def _maybe_wrap_stack_fn(stack_fn, ex_factory):
        """The stack_fn for SizingExecutorFactory requires two outputs.

        If required, we will wrap the stack_fn and provide a whimsy value as the
        second return value.

        Args:
          stack_fn: The original stack_fn
          ex_factory: A class which inherits from ExecutorFactory.

        Returns:
          A stack_fn that might additionally return a list as the second value.
        """
        if ex_factory == python_executor_stacks.SizingExecutorFactory:
            return lambda x: (stack_fn(x), [])
        else:
            return stack_fn

    def _stack_fn(x):
      del x  # Unused
      return eager_tf_executor.EagerTFExecutor()

    ex_factory = python_executor_stacks.ResourceManagingExecutorFactory
    maybe_wrapped_stack_fn = _maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    '''

    factory = python_executor_stacks.local_executor_factory(default_num_clients=3)

    server_utils.run_server(factory, 10, 30000, None, None)


if __name__ == '__main__':
    app.run(main)