import re
import itertools

import rdkit.Chem as rkc
import rdkit.Chem.Descriptors as rkcde

import utils.chem as uc


class SlicedMol:

    def __init__(self, scaffold, decorations):
        """
        Represents a molecule as a scaffold and the decorations associated with each attachment point.
        :param scaffold: A Mol object with the scaffold.
        :param decorations: Either a list or a dict with the decorations as Mol objects.
        """
        self.scaffold = uc.copy_mol(scaffold)

        if isinstance(decorations, list):
            decorations = [uc.copy_mol(dec) for dec in decorations]
            nums = [get_first_attachment_point(dec) for dec in decorations]
            self.decorations = {num: remove_attachment_point_numbers(dec)
                                for num, dec in zip(nums, decorations)}
        else:
            self.decorations = {num: remove_attachment_point_numbers(uc.copy_mol(dec))
                                for num, dec in decorations.items()}

        self._normalize()

    def __eq__(self, other_sliced_mol):
        return self.to_smiles() == other_sliced_mol.to_smiles()

    def __hash__(self):
        smi = self.to_smiles()
        return tuple([smi[0], *(smi[1].items())]).__hash__()

    def _normalize(self):
        """
        Normalizes the scaffold, given that the canonicalization algorithm uses the atom map number to canonicalize.
        """
        for atom in self.scaffold.GetAtoms():
            if atom.HasProp("molAtomMapNumber") and atom.GetSymbol() == ATTACHMENT_POINT_TOKEN:
                num = atom.GetProp("molAtomMapNumber")
                atom.ClearProp("molAtomMapNumber")
                atom.SetProp("_idx", num)

        _ = to_smiles(self.scaffold)
        atom_ordering = eval(self.scaffold.GetProp("_smilesAtomOutputOrder"))  # pylint: disable= eval-used

        curr2can = {}
        curr_idx = 0
        for atom_idx in atom_ordering:
            atom = self.scaffold.GetAtomWithIdx(atom_idx)
            if atom.HasProp("_idx") and atom.GetSymbol() == ATTACHMENT_POINT_TOKEN:
                num = int(atom.GetProp("_idx"))
                atom.ClearProp("_idx")
                atom.SetProp("molAtomMapNumber", str(curr_idx))
                curr2can[num] = curr_idx
                curr_idx += 1
        self.decorations = {curr2can[num]: dec for num, dec in self.decorations.items()}

    def to_smiles(self, variant="canonical"):
        """
        Calculates the SMILES representation of the given variant of the scaffold and decorations.
        :param variant: SMILES variant to use (see to_smiles)
        :return: A tuple with the SMILES of the scaffold and a dict with the SMILES of the decorations.
        """
        return (to_smiles(self.scaffold, variant=variant),
                {num: to_smiles(dec, variant=variant) for num, dec in self.decorations.items()})


ATTACHMENT_POINT_TOKEN = "*"
ATTACHMENT_POINT_NUM_REGEXP = r"\[{}:(\d+)\]".format(re.escape(ATTACHMENT_POINT_TOKEN))
ATTACHMENT_POINT_REGEXP = r"(?:{0}|\[{0}[^\]]*\])".format(re.escape(ATTACHMENT_POINT_TOKEN))
ATTACHMENT_POINT_NO_BRACKETS_REGEXP = r"(?<!\[){}".format(re.escape(ATTACHMENT_POINT_TOKEN))

ATTACHMENT_SEPARATOR_TOKEN = "|"

SLICE_SMARTS = {
    "hr": [
        "[*]!@-[*]"
    ],
    "recap": [
        "[C;$(C=O)]!@-N",  # amides and urea
        "[C;$(C=O)]!@-O",  # esters
        "C!@-[N;!$(NC=O)]",  # amines
        "C!@-[O;!$(NC=O)]",  # ether
        "[CX3]!@=[CX3]",  # olefin
        "[N+X4]!@-C",  # quaternary nitrogen
        "n!@-C",  # aromatic N - aliphatic C
        "[$([NR][CR]=O)]!@-C",  # lactam nitrogen - aliphatic carbon
        "c!@-c",  # aromatic C - aromatic C
        "N!@-[$(S(=O)=O)]"  # sulphonamides
    ]
}
SLICE_SMARTS = {name: [rkc.MolFromSmarts(sma) for sma in smarts] for name, smarts in SLICE_SMARTS.items()}


class SliceEnumerator:

    def __init__(self, slice_smarts, scaffold_conditions=None, decoration_conditions=None):
        """
        Class to enumerate slicings given certain conditions.
        :param slice_smarts: A list of SMARTS rules to cut molecules (see SLICE_SMARTS for an example).
        :param scaffold_conditions: Conditions to use when filtering scaffolds obtained from slicing molecules (see FragmentFilter).
        :param decoration_conditions: Conditions to use when filtering decorations obtained from slicing molecules.
        """
        self.slice_smarts = slice_smarts
        self._scaffold_filter = FragmentFilter(scaffold_conditions)
        self._decoration_filter = FragmentFilter(decoration_conditions)

    def count(self, mol):
        """
        Count the number of possible slicings in a given molecule.
        :param mol: Molecule to check.
        :return : An integer with the number of possible cuts.
        """
        return len(self._get_matches(mol))

    def enumerate(self, mol, cuts):
        """
        Enumerates all possible combination of slicings of a molecule given a number of cuts.
        :param mol: A mol object with the molecule to slice.
        :param cuts: The number of cuts to perform.
        :return : A list with all the possible (scaffold, decorations) pairs as SlicedMol objects.
        """
        matches = self._get_matches(mol)
        sliced_mols = set()
        for atom_pairs_to_cut in itertools.combinations(matches, cuts):
            to_cut_bonds = list(sorted(mol.GetBondBetweenAtoms(aidx, oaidx).GetIdx()
                                       for aidx, oaidx in atom_pairs_to_cut))
            attachment_point_idxs = [(i, i) for i in range(len(to_cut_bonds))]
            cut_mol = rkc.FragmentOnBonds(mol, bondIndices=to_cut_bonds, dummyLabels=attachment_point_idxs)
            for atom in cut_mol.GetAtoms():
                if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN:
                    num = atom.GetIsotope()
                    atom.SetIsotope(0)
                    atom.SetProp("molAtomMapNumber", str(num))

            cut_mol.UpdatePropertyCache()
            fragments = rkc.GetMolFrags(cut_mol, asMols=True, sanitizeFrags=True)

            # detect whether there is one fragment with as many attachment points as cuts (scaffold)
            # the rest are decorations
            if cuts == 1:
                sliced_mols.add(SlicedMol(fragments[0], [fragments[1]]))
                sliced_mols.add(SlicedMol(fragments[1], [fragments[0]]))
            else:
                scaffold = None
                decorations = []
                for frag in fragments:
                    num_att = len([atom for atom in frag.GetAtoms() if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN])
                    if num_att == cuts and not scaffold:
                        scaffold = frag
                    else:
                        decorations.append(frag)
                if scaffold:
                    sliced_mols.add(SlicedMol(scaffold, decorations))

        return list(filter(self._filter, sliced_mols))

    def _filter(self, sliced_mol):
        return self._scaffold_filter.filter(sliced_mol.scaffold) \
            and all(self._decoration_filter.filter(dec) for dec in sliced_mol.decorations.values())

    def _get_matches(self, mol):
        matches = set()
        for smarts in self.slice_smarts:
            matches |= set(tuple(sorted(match)) for match in mol.GetSubstructMatches(smarts))
        return list(matches)


class FragmentFilter:

    CONDITIONS_FUNC = {
        "hac": rkcde.HeavyAtomCount,  # pylint: disable=no-member
        "mol_weight": rkcde.MolWt,
        "clogp": rkcde.MolLogP,  # pylint: disable=no-member
        "hbd": rkcde.NumHDonors,  # pylint: disable=no-member
        "hba": rkcde.NumHAcceptors,  # pylint: disable=no-member
        "rotatable_bonds": rkcde.NumRotatableBonds,  # pylint: disable=no-member
        "num_rings": rkcde.RingCount  # pylint: disable=no-member
    }

    ATTACHMENT_POINT_MOL = rkc.MolFromSmiles("[{}]".format(ATTACHMENT_POINT_TOKEN))

    def __init__(self, conditions=None):
        """
        Initializes a fragment filter given the conditions.
        :param conditions: Conditions to use. When None is given, everything is valid.
        """
        if conditions is None:
            conditions = {}
        self.conditions = conditions

    def filter(self, mol):
        """
        Filters a molecules.
        :param mol: A molecule as a Mol object.
        :return: A boolean whether the molecule is valid or not.
        """
        return self._check_attachment_points(mol) and self._check_filters(mol)

    def _check_attachment_points(self, mol):
        return all(atom.GetDegree() == 1 for atom in mol.GetAtoms() if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN)

    def _check_filters(self, mol):
        # remove attachment points
        # dec = rkc.DeleteSubstructs(mol, self.ATTACHMENT_POINT_MOL)
        for name, comp_val in self.conditions.items():
            value = self.CONDITIONS_FUNC[name](mol)
            if isinstance(comp_val, dict):
                if "min" in comp_val:
                    if value < comp_val["min"]:
                        return False
                if "max" in comp_val:
                    if value > comp_val["max"]:
                        return False
            elif value != comp_val:
                return False
        return True


def _add_attachment_point_num(atom, idx):
    idxs = []
    if atom.HasProp("molAtomMapNumber"):
        idxs = atom.GetProp("molAtomMapNumber").split(",")
    idxs.append(str(idx))
    idxs = sorted(list(set(idxs)))
    atom.SetProp("molAtomMapNumber", ",".join(idxs))


def join(scaffold_smi, decoration_smi, keep_label_on_atoms=False):
    """
    Joins a SMILES scaffold with a decoration. They must be labelled.
    :param scaffold_smi: SMILES of the scaffold.
    :param decoration_smi: SMILES of the decoration.
    :param keep_label_on_atoms: Add the labels to the atoms after attaching the molecule.
    This is useful when debugging, but it can give problems.
    :return: A Mol object of the joined scaffold.
    """
    scaffold = uc.to_mol(scaffold_smi)
    decoration = uc.to_mol(decoration_smi)

    if scaffold and decoration:
        # obtain id in the decoration
        try:
            attachment_points = [atom.GetProp("molAtomMapNumber") for atom in decoration.GetAtoms()
                                 if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN]
            if len(attachment_points) != 1:
                return None  # more than one attachment point...
            attachment_point = attachment_points[0]
        except KeyError:
            return None

        combined_scaffold = rkc.RWMol(rkc.CombineMols(decoration, scaffold))
        attachments = [atom for atom in combined_scaffold.GetAtoms()
                       if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN and
                       atom.HasProp("molAtomMapNumber") and atom.GetProp("molAtomMapNumber") == attachment_point]
        if len(attachments) != 2:
            return None  # something weird

        neighbors = []
        for atom in attachments:
            if atom.GetDegree() != 1:
                return None  # the attachment is wrongly generated
            neighbors.append(atom.GetNeighbors()[0])

        bonds = [atom.GetBonds()[0] for atom in attachments]
        bond_type = rkc.BondType.SINGLE
        if any(bond for bond in bonds if bond.GetBondType() == rkc.BondType.DOUBLE):
            bond_type = rkc.BondType.DOUBLE

        combined_scaffold.AddBond(neighbors[0].GetIdx(), neighbors[1].GetIdx(), bond_type)
        combined_scaffold.RemoveAtom(attachments[0].GetIdx())
        combined_scaffold.RemoveAtom(attachments[1].GetIdx())

        if keep_label_on_atoms:
            for neigh in neighbors:
                _add_attachment_point_num(neigh, attachment_point)

        scaffold = combined_scaffold.GetMol()
        try:
            rkc.SanitizeMol(scaffold)
        except ValueError:  # sanitization error
            return None
    else:
        return None

    return scaffold


def join_first_attachment(scaffold_smi, decoration_smi):
    """
    Joins a SMILES scaffold with one decoration.
    :param scaffold_smi: SMILES of the scaffold.
    :param decoration_smi: SMILES of the decoration.
    :return : A Mol object of the joined scaffold.
    """
    new_scaffold_smi = add_first_attachment_point_number(scaffold_smi, 0)
    new_decoration_smi = add_first_attachment_point_number(decoration_smi, 0)
    return join(new_scaffold_smi, new_decoration_smi)


def join_joined_attachments(scaffold_smi, decorations_smi):
    decorations_smi = [add_first_attachment_point_number(dec, i)
                       for i, dec in enumerate(decorations_smi.split(ATTACHMENT_SEPARATOR_TOKEN))]
    scaffold_smi = add_attachment_point_numbers(scaffold_smi)
    num_att_points = len(get_attachment_points(scaffold_smi))
    if len(decorations_smi) != num_att_points:
        return None

    mol = uc.to_mol(scaffold_smi)
    for dec in decorations_smi:
        mol = join(to_smiles(mol), dec)
        if not mol:
            return None
    return mol


def remove_attachment_point_numbers(mol_or_smi):
    """
    Removes the numbers for the attachment points throughout the molecule.
    :param mol_or_smi: SMILES string or mol object to convert.
    :return : A converted SMILES string.
    """
    if isinstance(mol_or_smi, rkc.Mol):
        for atom in mol_or_smi.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
        return mol_or_smi
    return re.sub(ATTACHMENT_POINT_NUM_REGEXP, "[{}]".format(ATTACHMENT_POINT_TOKEN), mol_or_smi)


def add_attachment_point_numbers(mol_or_smi, canonicalize=True):
    """
    Adds the numbers for the attachment points throughout the molecule.
    :param mol_or_smi: SMILES string to convert.
    :param canonicalize: Canonicalize the SMILES so that the attachment points are always in the same order.
    :return : A converted SMILES string.
    """
    if isinstance(mol_or_smi, str):
        smi = mol_or_smi
        if canonicalize:
            smi = to_smiles(uc.to_mol(mol_or_smi))
        # only add numbers ordered by the SMILES ordering
        num = -1

        def _ap_callback(_):
            nonlocal num
            num += 1
            return "[{}:{}]".format(ATTACHMENT_POINT_TOKEN, num)

        return re.sub(ATTACHMENT_POINT_REGEXP, _ap_callback, smi)
    else:
        mol = mol_or_smi
        if canonicalize:
            mol = uc.to_mol(to_smiles(mol))
        idx = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN:
                atom.SetProp("molAtomMapNumber", str(idx))
                idx += 1
        return to_smiles(mol)


def add_brackets_to_attachment_points(smi):
    """
    Adds brackets to the attachment points (if they don't have them).
    :param smi: SMILES string.
    :return: A SMILES string with attachments with brackets.
    """
    return re.sub(ATTACHMENT_POINT_NO_BRACKETS_REGEXP, "[{}]".format(ATTACHMENT_POINT_TOKEN), smi)


def add_first_attachment_point_number(smi, num):
    """
    Changes/adds a number to the first attachment point.
    :param smi: SMILES string with the molecule.
    :param num: Number to add.
    :return: A SMILES string with the number added.
    """
    return re.sub(ATTACHMENT_POINT_REGEXP, "[{}:{}]".format(ATTACHMENT_POINT_TOKEN, num), smi, count=1)


def get_first_attachment_point(mol_or_smi):
    """
    Obtains the number of the first attachment point.
    :param mol_or_smi: A Mol object or a SMILES string
    :return: The number of the first attachment point.
    """
    return get_attachment_points(mol_or_smi)[0]


def get_attachment_points(mol_or_smi):
    """
    Gets all attachment points regardless of the format.
    :param mol_or_smi: A Mol object or a SMILES string
    :return : A list with the numbers ordered by appearance.
    """
    if isinstance(mol_or_smi, rkc.Mol):
        return [int(atom.GetProp("molAtomMapNumber")) for atom in mol_or_smi.GetAtoms()
                if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN and atom.HasProp("molAtomMapNumber")]
    return [int(match.group(1)) for match in re.finditer(ATTACHMENT_POINT_NUM_REGEXP, mol_or_smi)]


def num_attachment_points(mol_or_smi):
    """
    Returns the number of attachment points of a given scaffold.
    :param mol_or_smi: A Mol object or a SMILES string.
    :return : The number of attachment points.
    """
    if isinstance(mol_or_smi, rkc.Mol):
        return len([atom for atom in mol_or_smi.GetAtoms() if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN])
    return len(re.findall(ATTACHMENT_POINT_REGEXP, mol_or_smi))


def to_smiles(mol, variant="canonical"):
    """
    Converts to a SMILES string and adds brackets to the attachment points.
    :param mol: A Mol object.
    :param variant: SMILES variant used.
    :return : A SMILES strings.
    """
    smi = uc.to_smiles(mol, variant)
    conv_smi = None
    if smi:
        conv_smi = add_brackets_to_attachment_points(smi)
    return conv_smi
