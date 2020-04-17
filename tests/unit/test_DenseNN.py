from collections import namedtuple
from cross_entropy.DenseNN import DenseNN
import pytest
import torch


Model = namedtuple('Model', ['n_inputs', 'n_hidden', 'wide_connection'])

model_params = []
n_inputs = (3, 17, 42)
n_hidden = (10, 2, 30)
wide_connection = (True, True, False)
for inputs in zip(n_inputs, n_hidden, wide_connection):
    model_params.append(Model(inputs[0], inputs[1], inputs[2]))
@pytest.fixture(params=model_params)
def model_params(request):
    return request.param

neg_model_params = []
neg_input = (-1, 24, 0, 30, -5)
neg_hidden = (10, -2, 42, 0, 4)
neg_wide_conn = (True, True, False, True, False)
for inputs in zip(neg_input, neg_hidden, neg_wide_conn):
    neg_model_params.append(Model(inputs[0], inputs[1], inputs[2]))
@pytest.fixture(params=neg_model_params)
def neg_model_params(request):
    return request.param


@pytest.mark.DenseNN
@pytest.mark.creation
def test_DenseNN_creation(model_params):
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
def test_negative_creation_params(neg_model_params):
    with pytest.raises(ValueError):
        model = DenseNN(neg_model_params.n_inputs, neg_model_params.n_hidden,
                        neg_model_params.wide_connection)
    

def test_DenseNN_forward():
    """
    tests the general structure of the output of a DenseNN object
    """
    # TODO: put these parameters into a model fixture
    n_inputs = 3
    n_hidden = 10
    wide_connection = True
    
    model = DenseNN(n_inputs=n_inputs, n_hidden=n_hidden, wide_connection=wide_connection)

    # TODO: put these parameters into a DenseNN inputs fixture
    inputs = torch.rand(10, 3)
    with torch.no_grad():
        output = model(inputs)

    assert output.shape == torch.Size([10, 10])
