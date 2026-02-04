import argparse
import torch
from uninas import UNIModel, UNIModelCfg, create_new_model

def parse_args():
    parser = argparse.ArgumentParser(description='UNINas Search')

    # Evolution parameters
    parser.add_argument('--num-iter', type=int, default=5, metavar='N',
                        help='Number of evolution iterations')
    parser.add_argument('--init-model', type=str, default='[["E", "E"], ["E", "R", "R"], ["T", "T", "T", "T", "T"], ["E", "R"]]',
                        metavar='N', help='Initial structure string.')
    parser.add_argument('--flops-min', type=int, default=0, metavar='N',
                        help='Min. number of FLOPs for a model')
    parser.add_argument('--flops-max', type=int, default=int(30e9), metavar='N',
                        help='Max. number of FLOPs for a model')
    parser.add_argument('--params-min', type=int, default=0, metavar='N',
                        help='Min. number of params for a model')
    parser.add_argument('--params-max', type=int, default=int(40e6), metavar='N',
                        help='Max. number of params for a model')
    parser.add_argument('--n-changes', type=int, default=1,
                        help='Number of eliminations/addings per iteration')
    parser.add_argument('--n-patience', type=int, default=5,
                        help='Number of attempts to create a new model within budget')
    parser.add_argument('--p-eliminate', type=float, default=0.3,
                        help='Probability of eliminating a node')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = UNIModel(UNIModelCfg(model_str=args.init_model)).to(device)
    criterion = torch.nn.MSELoss()

    # Forward/backward pass with dummy data
    inputs = torch.rand(4, 3, 224, 224, device=device)
    targets = torch.randn(4, 1000, device=device)
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()

    for iteration in range(args.num_iter):
        # Create mutated model
        new_model = create_new_model(
            model,
            flops_min=args.flops_min,
            flops_max=args.flops_max,
            params_min=args.params_min,
            params_max=args.params_max,
            n_patience=args.n_patience,
            n_changes=args.n_changes,
            p_eliminate=args.p_eliminate
        )

        # Skip iteration if mutation failed
        if new_model is None:
            continue

        model = new_model.to(device)

        # Forward/backward pass with dummy data
        inputs = torch.rand(4, 3, 224, 224, device=device)
        targets = torch.randn(4, 1000, device=device)
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()

        print(f"Iteration {iteration + 1}/{args.num_iter}: loss={loss.item():.4f}")

    # Model to string
    model_str = model.to_string()

    # Model from string
    model_from_str = UNIModel(UNIModelCfg.from_string(model_str))

    # Test model from string
    output = model_from_str(inputs)
    loss = criterion(output, targets)
    loss.backward()

    # Compare models
    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model_from_str.parameters()))


if __name__ == "__main__":
    main()
