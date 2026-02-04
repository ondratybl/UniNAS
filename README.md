# uninas  
  
This repository provides a framework to create instances of UNINas search space and walk through the space.    
  
## Testing  

### Initial string format  
The string is structured by stages, separated by `//`. Each stage contains a sequence of blocks, separated by `/`. We support the following number of stages and blocks: `(2, 3, 5, 2)`, e.g. `MODEL_STRING='E/R//T/T/T//T/T/T/T/T//T/T'`

In this example, the respective characters denote blocks corresponding to:  
- `T` Transformer  
- `E` EfficientNet  
- `R` ResNet

Instead of these variants (`T`, `E`, `R`), any plausible structure of the following form can be used:

`subblock1[XXX]subblock1subblock2[YYY]`,

where `XXX` and `YYY` is a recursive graph representation obtained via each Node's method .to_string(). An example corresponding to `MODEL_STRING='E/R//T/T/T//T/T/T/T/T//T/T'` can be found in `model_string.txt`

  
```bash  
python test.py --init-model <MODEL_STRING>