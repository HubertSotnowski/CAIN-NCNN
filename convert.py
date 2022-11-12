from cain.cain import CAIN
import torch
import argparse
my_parser = argparse.ArgumentParser(description=" ")
my_parser.add_argument("--input", metavar="--input", type=str, help="input model")
args = my_parser.parse_args()

import os
model=CAIN(3)
model.load_state_dict(torch.load(args.input), strict=False)
input_names = ["in1", "in2"]
output_names = ["output_frame"]
f1=torch.rand((1,3,256,256))
f2=torch.rand((1,3,256,256))
x=(f1,f2)
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "cain.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = input_names,   # the model's input names
                  output_names = output_names) # the model's output names)
os.system("python3 -m onnxsim cain.onnx cain-sim.onnx")


