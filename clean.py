import onnx
import argparse


def main(args):
    if not args.keep_node_name:
        raise ValueError("Provide at least one node name to keep")
    
    model = onnx.load(args.onnx_file_path)
    
    # Print original outputs
    print("\t====== Original outputs ======")
    for output in model.graph.output:
        print(output)

    # Keep only the first output
    nodes = []
    for output in model.graph.output:
        if output.name in args.keep_node_name:
            nodes.append(output)

    # Remove all existing outputs by reconstructing a new list
    del model.graph.output[:]
    model.graph.output.extend(nodes)
    
    print("\t====== Remaining outputs ======")
    for output in model.graph.output:
        print(output)
        
    # Save modified model
    onnx.save(model, args.onnx_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_file_path", required=True, type=str)
    parser.add_argument("--keep_node_name", nargs='+', type=str, default=None)
    args = parser.parse_args()
    
    main(args)