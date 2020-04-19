from collections import namedtuple
from cross_entropy.DenseNN import DenseNN
import pytest
import torch


Model = namedtuple('Model', ['n_inputs', 'n_hidden', 'wide_connection'])

model_params = []
model_params_ids = []
n_inputs = (3, 17, 42)
n_hidden = (10, 2, 30)
wide_connection = (True, True, False)
for inputs in zip(n_inputs, n_hidden, wide_connection):
    model_params.append(Model(*inputs))
    model_params_ids.append('in={}, hidden={}, wide={}'.format(*inputs))
@pytest.fixture(params=model_params, ids=model_params_ids)
def model_params(request):
    return request.param

neg_model_params = []
neg_model_params_ids = []
neg_input = (-1, 24, 0, 30, -5)
neg_hidden = (10, -2, 42, 0, 4)
neg_wide_conn = (True, True, False, True, False)
for inputs in zip(neg_input, neg_hidden, neg_wide_conn):
    neg_model_params.append(Model(*inputs))
    neg_model_params_ids.append('in={}, hidden={}, wide={}'.format(*inputs))
@pytest.fixture(params=neg_model_params, ids=neg_model_params_ids)
def neg_model_params(request):
    return request.param


@pytest.mark.DenseNN
@pytest.mark.creation
def test_densenn_creation(model_params):
    """
    tests the structure of the DenseNN object
    """
    model = DenseNN(model_params.n_inputs, model_params.n_hidden, 
                    model_params.wide_connection)
    assert model.layer1.in_features == model_params.n_inputs
    assert model.layer1.out_features == model_params.n_hidden
    assert model.wide_connection == model_params.wide_connection
    if model.wide_connection:
        assert model.layer2.in_features == model_params.n_inputs + model_params.n_hidden
    else:
        assert model.layer2.in_features == model_params.n_hidden
    assert model.layer2.out_features == model_params.n_hidden


@pytest.mark.DenseNN
@pytest.mark.creation
@pytest.mark.negative_input
def test_densenn_negative_creation(neg_model_params):
    with pytest.raises(ValueError):
        model = DenseNN(neg_model_params.n_inputs, neg_model_params.n_hidden,
                        neg_model_params.wide_connection)
    

@pytest.mark.DenseNN
def test_densenn_forward(model_params):
    """
    tests the general structure of the output of a DenseNN object
    """
    model = DenseNN(model_params.n_inputs, model_params.n_hidden, 
                    model_params.wide_connection)

    n_samples = torch.randint(low=1, high=1000, size=(1,))
    inputs = torch.rand(n_samples, model_params.n_inputs)
    with torch.no_grad():
        output = model(inputs)

    assert output.shape == torch.Size([n_samples, model_params.n_hidden])
