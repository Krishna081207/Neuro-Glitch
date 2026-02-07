import torch
from models import predict_journal, fine_tune_model

def test_tensor_operations():
    tensor_a = torch.rand(2, 3)
    tensor_b = torch.rand(3, 2)
    result = torch.matmul(tensor_a, tensor_b)
    assert result.shape == (2, 2)
    print("Tensor multiplication test passed.")

def test_predict_journal():
    text = "I feel like I'm drowning in my work and can't see a way out."
    pred = predict_journal(text)
    assert pred in [0, 1]
    print("Journal prediction test passed.")

def test_fine_tune():
    # This is a placeholder test; actual test requires a real dataset
    result = fine_tune_model('mental_health.csv')
    print("Fine-tune test result:", result)

if __name__ == "__main__":
    test_tensor_operations()
    test_predict_journal()
    test_fine_tune()
