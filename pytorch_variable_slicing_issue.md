

```python
import torch
from torch.autograd import Variable
import numpy as np
```


```python
out=np.array([[5.5737e-08,1.2218e-07]],dtype=np.float32)
out = Variable(torch.from_numpy(out).cuda())
print(out)
```

    Variable containing:
    1.00000e-07 *
      0.5574  1.2218
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]




```python
zero_one_mat = np.zeros((1,2),dtype=np.float32)
zero_one_mat[0,0] = 1.
zero_one_mat = torch.from_numpy(zero_one_mat).cuda()
print(zero_one_mat)
```


     1  0
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]




```python
print('wrong result:')

out[0,:].data = torch.mul(out[0,:].data,zero_one_mat)
print(out.data)

print('**************')
print('correct result:')
print(torch.mul(out[0,:].data,zero_one_mat))

print(torch.mul(out.data[0,:],zero_one_mat))

out.data[0,:] = torch.mul(out[0,:].data,zero_one_mat)
print(out.data)

out=np.array([[5.5737e-08,1.2218e-07]],dtype=np.float32)
out = Variable(torch.from_numpy(out).cuda())
out.data[0,:] = torch.mul(out.data[0,:],zero_one_mat)
print(out.data)

out=np.array([[5.5737e-08,1.2218e-07]],dtype=np.float32)
out = Variable(torch.from_numpy(out).cuda())
torch.mul(out[0,:].data,zero_one_mat,out=out[0,:].data)
print(out.data)

out=np.array([[5.5737e-08,1.2218e-07]],dtype=np.float32)
out = Variable(torch.from_numpy(out).cuda())
torch.mul(out.data[0,:],zero_one_mat,out=out[0,:].data)
print(out.data)

out=np.array([[5.5737e-08,1.2218e-07]],dtype=np.float32)
out = Variable(torch.from_numpy(out).cuda())
torch.mul(out.data[0,:],zero_one_mat,out=out.data[0,:])
print(out.data)

out=np.array([[5.5737e-08,1.2218e-07]],dtype=np.float32)
out = Variable(torch.from_numpy(out).cuda())
torch.mul(out[0,:].data,zero_one_mat,out=out.data[0,:])
print(out.data)
```

    wrong result:

    1.00000e-07 *
      0.5574  1.2218
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]
    
    **************
    correct result:
    
    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


    1.00000e-08 *
      5.5737  0.0000
    [torch.cuda.FloatTensor of size 1x2 (GPU 0)]


