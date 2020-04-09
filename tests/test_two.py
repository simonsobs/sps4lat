from sps4lat import covariance

def test_add():
    a = 3
    b = 4
    assert covariance.multiply(a, b) == (a * b)
