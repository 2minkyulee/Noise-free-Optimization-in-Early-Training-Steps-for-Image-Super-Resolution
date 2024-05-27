# Adding Losses

### Instructions
```text
1. Add a file in /PROJ_ROOT/loss/LOSS_NAME.py
2. Define the loss with class (nn.Module) with any name (does not need to match the file name).
3. Modify loss.__init__.py (search for @tag [Loss Definitions])
4. Modify the loss_string given in arg-parsers. 
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

