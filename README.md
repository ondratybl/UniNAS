# uninas

This repository provides a framework for creating model instances from the **UNINas search space** and exploring (walking through) that space programmatically. It supports JSON-defined architectures, conversion between JSON and model objects, and search workflows over valid configurations.

---

## Model JSON Format

The model description JSON defines the structural and hyperparameter configuration of a model architecture within the UniNAS search space.

**Reference papers:**

- UniNAS search space — https://arxiv.org/abs/2510.06035  
- CoAtNet — https://arxiv.org/abs/2106.04803

---

### Example — CoAtNet-like Configuration

```json
{
  "img_size": 224,
  "num_classes": 1000,
  "drop_rate": 0.0,
  "embed_dim": [96, 192, 384, 768],
  "depths": [2, 3, 5, 2],
  "model_str": "[[\"E\", \"E\"], [\"E\", \"E\", \"E\"], [\"T\", \"T\", \"T\", \"T\", \"T\"], [\"T\", \"T\"]]",
  "stem_width": [32, 64]
}
```

### Field Meaning

The following fields define standard architectural parameters:

- `img_size` — input image resolution (square)  
- `num_classes` — number of output classes  
- `drop_rate` — dropout rate  
- `embed_dim` — per-stage embedding/channel dimensions  
- `depths` — number of blocks per stage  
- `stem_width` — channel widths of stem layers  

The `model_str` field encodes the **stage-by-stage, block-by-block layout** as a stringified nested array.

In the example above `"[[\"E\", \"E\"], [\"E\", \"E\", \"E\"], [\"T\", \"T\", \"T\", \"T\", \"T\"], [\"T\", \"T\"]]"` stands for:

- Stage 1: 2 EfficientNet-like blocks  
- Stage 2: 3 EfficientNet-like blocks  
- Stage 3: 5 Transformer-like blocks  
- Stage 4: 2 Transformer-like blocks  

---

### Block Type Codes in `model_str`

Each character represents a block type:

- `T` — Transformer block  
- `E` — EfficientNet-like block  
- `R` — ResNet-like block  

Moreover, instead of single characters, we can specify a fully custom block from the UniNAS search space via a structured block definition: `{"subblock1": XXX, "subblock2": YYY}`, where `XXX` and `YYY` are recursive graph representations produced via each Node’s `.to_string()` method.

Additional examples can be found in `examples/`
- `model_coatnet.json` -- CoAtNet, 
- `model_efficientnet.json` -- EfficientNet, 
- `model_resnet.json` -- ResNet,
- `model_transformer.json` -- Transformer
- `model_example.json` -- Custom model 

### Code
json-to-model (and back) is performed via

```python 
from uninas import UNIModel, UNIModelCfg
import json

# Json to model
model_path_in = 'examples/model_example.json'
model_path_out = 'examples/model_example_out.json'
with open(model_path_in, "r") as f:
    model_string = f.read()
model_cfg = UNIModelCfg.from_string(model_string)
model = UNIModel(model_cfg)

# Model to json
model_str = model.to_string()
with open(model_path_out, "w") as f:
    json.dump(json.loads(model_str), f, indent=2)
```
You can load an initial model from JSON, perform a search in the UniNAS search space with specific search parameters, and save the resulting model configuration:
```bash
python examples/test.py --model-init <MODEL_JSON_PATH>