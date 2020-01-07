"""
RDKit util functions.
"""
import random
import gzip

import rdkit.Chem as rkc


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()


def read_smi_file(file_path, ignore_invalid=True, num=-1):
    """
    Reads a SMILES file.
    :param file_path: Path to a SMILES file.
    :param ignore_invalid: Ignores invalid lines (empty lines)
    :param num: Parse up to num SMILES.
    :return: A list with all the SMILES.
    """
    return map(lambda fields: fields[0], read_csv_file(file_path, ignore_invalid, num))


def read_csv_file(file_path, ignore_invalid=True, num=-1, num_fields=0):
    """
    Reads a SMILES file.
    :param file_path: Path to a CSV file.
    :param ignore_invalid: Ignores invalid lines (empty lines)
    :param num: Parse up to num rows.
    :param num_fields: Number of fields to extract (from left to right).
    :return: An iterator with the rows.
    """
    with open_file(file_path, "rt") as csv_file:
        for i, row in enumerate(csv_file):
            if i == num:
                break
            fields = row.rstrip().split("\t")
            if fields:
                if num_fields > 0:
                    fields = fields[0:num_fields]
                yield fields
            elif not ignore_invalid:
                yield None


def open_file(path, mode="r", with_gzip=False):
    """
    Opens a file depending on whether it has or not gzip.
    :param path: Path where the file is located.
    :param mode: Mode to open the file.
    :param with_gzip: Open as a gzip file anyway.
    """
    open_func = open
    if path.endswith(".gz") or with_gzip:
        open_func = gzip.open
    return open_func(path, mode)


def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smi:
        return rkc.MolFromSmiles(smi)


def to_smiles(mol, variant="canonical"):
    """
    Converts a Mol object into a canonical SMILES string.
    :param mol: Mol object.
    :return: A SMILES string.
    """
    if mol:
        if variant.startswith("random"):
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
            return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
        else:
            return rkc.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def copy_mol(mol):
    """
    Copies, sanitizes, canonicalizes and cleans a molecule.
    :param mol: A Mol object to copy.
    :return : Another Mol object copied, sanitized, canonicalized and cleaned.
    """
    return to_mol(to_smiles(mol))
