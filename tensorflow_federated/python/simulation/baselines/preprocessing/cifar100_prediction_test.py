# Copyright 2019, Google LLC.
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

import collections
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.simulation.baselines.preprocessing import cifar100_prediction


TEST_DATA = collections.OrderedDict(
    coarse_label=([tf.constant(1, dtype=tf.int64)]),
    image=([tf.zeros((32, 32, 3), dtype=tf.uint8)]),
    label=([tf.constant(1, dtype=tf.int64)]),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  def test_preprocess_fn_with_negative_epochs_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'num_epochs must be a positive integer'):
      cifar100_prediction.create_preprocess_fn(num_epochs=-2, batch_size=1)

  def test_raises_non_iterable_crop(self):
    with self.assertRaisesRegex(TypeError, 'crop_shape must be an iterable'):
      cifar100_prediction.create_preprocess_fn(
          num_epochs=1, batch_size=1, crop_shape=32)

  def test_raises_iterable_length_2_crop(self):
    with self.assertRaisesRegex(ValueError,
                                'The crop_shape must have length 3'):
      cifar100_prediction.create_preprocess_fn(
          num_epochs=1, batch_size=1, crop_shape=(32, 32))

  @parameterized.named_parameters(
      ('num_epochs_1_batch_size_1', 1, 1),
      ('num_epochs_4_batch_size_2', 4, 2),
      ('num_epochs_9_batch_size_3', 9, 3),
      ('num_epochs_12_batch_size_1', 12, 1),
      ('num_epochs_3_batch_size_5', 3, 5),
      ('num_epochs_7_batch_size_2', 7, 2),
  )
  def test_ds_length_is_ceil_num_epochs_over_batch_size(self, num_epochs,
                                                        batch_size):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = cifar100_prediction.create_preprocess_fn(
        num_epochs=num_epochs, batch_size=batch_size, shuffle_buffer_size=1)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        tf.cast(tf.math.ceil(num_epochs / batch_size), tf.int32))

  @parameterized.named_parameters(
      ('crop_shape_1_no_distort', (32, 32, 3), False),
      ('crop_shape_2_no_distort', (28, 28, 3), False),
      ('crop_shape_3_no_distort', (24, 26, 3), False),
      ('crop_shape_1_distort', (32, 32, 3), True),
      ('crop_shape_2_distort', (28, 28, 3), True),
      ('crop_shape_3_distort', (24, 26, 3), True),
  )
  def test_preprocess_fn_returns_correct_element(self, crop_shape,
                                                 distort_image):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = cifar100_prediction.create_preprocess_fn(
        num_epochs=1,
        batch_size=1,
        shuffle_buffer_size=1,
        crop_shape=crop_shape,
        distort_image=distort_image)
    preprocessed_ds = preprocess_fn(ds)
    expected_element_spec_shape = (None,) + crop_shape
    self.assertEqual(
        preprocessed_ds.element_spec,
        (tf.TensorSpec(shape=expected_element_spec_shape, dtype=tf.float32),
         tf.TensorSpec(shape=(None,), dtype=tf.int64)))

    expected_element_shape = (1,) + crop_shape
    element = next(iter(preprocessed_ds))
    expected_element = (tf.zeros(
        shape=expected_element_shape,
        dtype=tf.float32), tf.ones(shape=(1,), dtype=tf.int32))
    self.assertAllClose(self.evaluate(element), expected_element)

  def test_preprocess_is_no_op_for_normalized_image(self):
    crop_shape = (1, 1, 3)
    x = tf.constant([[[1.0, -1.0, 0.0]]])  # Has shape (1, 1, 3), mean 0
    x = x / tf.math.reduce_std(x)  # x now has variance 1
    simple_example = collections.OrderedDict(image=x, label=0)
    image_map = cifar100_prediction.build_image_map(crop_shape, distort=False)
    cropped_example = image_map(simple_example)

    self.assertEqual(cropped_example[0].shape, crop_shape)
    self.assertAllClose(x, cropped_example[0], rtol=1e-03)
    self.assertEqual(cropped_example[1], 0)


CIFAR100_LOAD_DATA = 'tensorflow_federated.python.simulation.datasets.cifar100.load_data'


class FederatedDatasetTest(tf.test.TestCase):

  @mock.patch(CIFAR100_LOAD_DATA)
  def test_preprocess_applied(self, mock_load_data):
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    mock_train = mock.create_autospec(client_data.ClientData)
    mock_test = mock.create_autospec(client_data.ClientData)
    mock_load_data.return_value = (mock_train, mock_test)

    _, _ = cifar100_prediction.get_federated_datasets()

    mock_load_data.assert_called_once()

    # Assert the training and testing data are preprocessed.
    self.assertEqual(mock_train.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())


class CentralizedDatasetTest(tf.test.TestCase):

  @mock.patch(CIFAR100_LOAD_DATA)
  def test_preprocess_applied(self, mock_load_data):
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    sample_ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)

    mock_train = mock.create_autospec(client_data.ClientData)
    mock_train.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_test = mock.create_autospec(client_data.ClientData)
    mock_test.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_load_data.return_value = (mock_train, mock_test)

    _, _ = cifar100_prediction.get_centralized_datasets()

    mock_load_data.assert_called_once()

    # Assert the validation ClientData isn't used, and the train and test
    # are amalgamated into datasets single datasets over all clients.
    self.assertEqual(mock_train.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
