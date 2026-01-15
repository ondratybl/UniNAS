from uninas import UNIModel, UNIModelCfg
import torch

# depths = (2, 3, 5, 2)
model_str = 'T-T/T-T-T/T-T-T-T-T/T-T'  # replace 'T' with any of 'E', 'R', 'T'
#model_str = 'E-E/E-E-E/T-T-T-T-T/T-T'
#model_str = 'E-E/E-E-E/E-E-E-E-E/E-E'
#model_str = 'R-R/R-R-R/R-R-R-R-R/R-R'

model = UNIModel(UNIModelCfg(model_str=model_str))
criterion = torch.nn.MSELoss()

# Forward/backward pass
inputs = torch.rand(4, 3, 224, 224)
y = torch.randn(4, 1000)
output = model(inputs)
loss = criterion(output, y)
loss.backward()

