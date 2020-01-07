#!/usr/bin/env python
#  coding=utf-8


import argparse
import functools

import tqdm

import models.model as mm
import models.actions as ma
import utils.log as ul
import utils.chem as uc


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Samples a model.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--input-scaffold-path", "-i",
                        help="Path to the input file with scaffolds in SMILES notation.", type=str, required=True)
    parser.add_argument("--output-smiles-path", "-o",
                        help="Path to the output file (if none given it will use stdout).", type=str)
    parser.add_argument("--with-nll", help="Store the NLL in a column after the SMILES.",
                        action="store_true", default=False)
    parser.add_argument("--batch-size", "-b",
                        help="Batch size (beware GPU memory usage) [DEFAULT: 128]", type=int, default=128)
    parser.add_argument("--use-gzip", help="Compress the output file (if set).", action="store_true", default=False)

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    model = mm.DecoratorModel.load_from_file(args.model_path, mode="eval")

    input_scaffolds = list(uc.read_smi_file(args.input_scaffold_path))
    if args.output_smiles_path:
        if args.use_gzip:
            args.output_smiles_path += ".gz"
        output_file = uc.open_file(args.output_smiles_path, "w+")
        write_func = functools.partial(output_file.write)
    else:
        output_file = tqdm.tqdm
        write_func = functools.partial(output_file.write, end="")

    sample_model = ma.SampleModel(model, args.batch_size)

    for scaff, dec, nll in ul.progress_bar(sample_model.run(input_scaffolds), total=len(input_scaffolds)):
        output_row = [scaff, dec]
        if args.with_nll:
            output_row.append("{:.8f}".format(nll))
        write_func("\t".join(output_row) + "\n")

    if args.output_smiles_path:
        output_file.close()


LOG = ul.get_logger(name="sample_from_model")
if __name__ == "__main__":
    main()
