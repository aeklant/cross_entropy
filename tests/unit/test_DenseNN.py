from cross_entropy.DenseNN import DenseNN
import torch # TODO: remove this once the input fixtures have been created elsewhere


def test_DenseNN_creation():
    """
    tests the structure of the DenseNN object
    """
    # TODO: put these parameters into a model fixture
    n_inputs = (3, 17, 42)
    n_hidden = (10, 2, 30)
    wide_connection = (True, True, False)
    
    model = DenseNN(n_inputs=n_inputs[0], n_hidden=n_hidden[0], 
                    wide_connection=wide_connection[0])
    assert model.layer1.in_features == 3
    assert model.layer1.out_features == 10
    assert model.layer2.in_features == 13
    assert model.layer2.out_features == 10
    assert model.wide_connection

    model = DenseNN(n_inputs=n_inputs[1], n_hidden=n_hidden[1],
                    wide_connection=wide_connection[1])
    assert model.layer1.in_features == 17
    assert model.layer1.out_features == 2
    assert model.layer2.in_features == 19
    assert model.layer2.out_features == 2
    assert model.wide_connection

    model = DenseNN(n_inputs=n_inputs[2], n_hidden=n_hidden[2],
                    wide_connection=wide_connection[2])
    assert model.layer1.in_features == 42
    assert model.layer1.out_features == 30
    assert model.layer2.in_features == 30
    assert model.layer2.out_features == 30
    assert not model.wide_connection

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
