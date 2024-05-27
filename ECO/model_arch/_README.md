# Adding Network Architectures

### Instructions
```text
1. Add a file in /PROJ_ROOT/model_arch/MODELNAME_arch.py
2. Define the network with class (nn.Module) with equal name as the file name, MODELNAME.
```

### Example
```python
# >>> ./model_arch/NewModelName_arch.py
class NewModelName(nn.Module):
    def __init__(self, *args, **kwargs):
        xxxxx
        xxxxx
        
    def forward(self, lr, others):
        xxxxx
        xxxxx

```

