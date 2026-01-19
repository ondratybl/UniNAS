# uninas  
  
This repository provides a framework to create instances of UNINas search space and walk through the space.    
  
## Testing  

### Initial string format  
Each character in the string represents a node type:  
- `T` Transformer block  
- `E` EfficientNet block  
- `R` ResNet block  
The string is structured by stages, separated by `/`. Each stage contains a sequence of nodes, separated by `-`. We support the following number of stages and blocks: `(2, 3, 5, 2)`, e.g. `MODEL_STRING='E-E/E-E-E/T-T-T-T-T/T-T'`
  
```bash  
python test.py --init-model <MODEL_STRING>