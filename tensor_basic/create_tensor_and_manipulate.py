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

temp = torch.tensor([[1,2],[3,4]])
print(temp.numpy()) #텐서를 numpy로 변환

temp = torch.tensor([[1,2],[3,4]], device =  'cuda:0')
print(temp.to('cpu').numpy()) #GPU싱의 텐서를 CPU텐서로 바꾸고 다시 numpy로 변환

'''
[[1 2]
 [3 4]]
[[1 2]
 [3 4]]
'''

temp = torch.FloatTensor([1,2,3,4,5,6,7])
print(temp[0], temp[1], temp[-1])
print("---------------------------------")
print(temp[2:5], temp[4:-1])
'''
tensor(1.) tensor(2.) tensor(7.)
---------------------------------
tensor([3., 4., 5.]) tensor([5., 6.])
'''

