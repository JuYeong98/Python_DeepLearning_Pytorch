import torch

print(torch.tensor([[1,2],[3,4]]))
print(torch.tensor([[1,2],[3,4]] , device = 'cuda:0'))
print(torch.tensor([[1,2],[3,4]], dtype = torch.float64))

'''
tensor([[1, 2],
        [3, 4]])
tensor([[1, 2],
        [3, 4]], device='cuda:0')
tensor([[1., 2.],
        [3., 4.]], dtype=torch.float64)
'''