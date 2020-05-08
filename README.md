Implementation of the SMILES-based scaffold decorator used in "SMILES-based deep generative scaffold decorator for de-novo drug design"
=======================================================================================================================================

This repository holds all the code used to create, train and sample a SMILES-based scaffold decorator described in [SMILES-based deep generative scaffold decorator for de-novo drug design](https://chemrxiv.org/articles/SMILES-Based_Deep_Generative_Scaffold_Decorator_for_De-Novo_Drug_Design/11638383). Additionally, it contains the code for pre-processing the training set, as explained in the manuscript. 

The scripts and folders are the following:

1) Python files in the main folder are all scripts. Run them with `-h` for usage information.
2) `./training_sets` folder: The two molecular sets used in the manuscript, separated between training and validations sets.
3) `./trained_models` folder: One already trained model for both the DRD2 and ChEMBL models.

Requirements
------------
The repository includes a Conda `environment.yml` file with the required libraries to run all the scripts. In some scripts Spark 2.4 is required (and thus Java 8) and by default should run in local mode without any issues. For more complex set-ups, please refer to the [Spark documentation](http://spark.apache.org/docs/2.4.3/). All models were tested on Linux with both a Tesla V-100 and a Geforce 2070. It should work just fine with other Linux setups and a mid-high range GPU.

Install
-------
A [Conda](https://conda.io/miniconda.html) `environment.yml` is supplied with all the required libraries.

~~~~
$> git clone <repo url>
$> cd <repo folder>
$> conda env create -f environment.yml
$> conda activate reinvent-scaffold-decorator
(reinvent-scaffold-decorator) $> ...
~~~~

From here the general usage applies.

General Usage
-------------
Several tools are supplied and are available as scripts in the main folder. Further information about the tool's arguments, please run it with `-h`. All output files are in tsv format (the separator is \t).

### Preprocessing the training sets

Any arbitrary molecular set has to be pre-processed before being used as a training set for a decorator model. This process is done in two steps:

1) Slice (`slice_db.py`): This script accepts as input a SMILES file and it exhaustively slices given a set of slicing rules (Hussain-Rea, RECAP), a maximum number of attachment points and a set of conditions (see `conditions.json.example` for more information). Rules can be easily extended and new sets of conditions added. This script can output a SMILES file with (scaffold, dec1;dec2;...) which can be used in the next step and/or a parquet file with additional information.
2) Create randomized SMILES (`create_randomized_smiles.py`): From the SMILES output of the first file, several randomized SMILES representations of the training set must be generated. Depending whether a single-step or a multi-step decorator model is wanted, different files are generated.

Notice that both scripts use Spark.

### Creating, training and sampling decorator models

This code enables training of both single-step and multi-step decorator models. The training process is exactly the same, only changing the training set pre-processing.

1) Create Model (`create_model.py`): Creates a blank model file.
2) Train Model (`train_model.py`): Trains the model with the specified parameters. Tensorboard data may be generated.
3) Sample Model (`sample_from_model.py`): Samples an already trained model for a given number of decorations given a list of scaffolds. It can also retrieve the log-likelihood in the process. Notice that this is not the preferred way of sampling the model (see below).
4) Calculate NLL (`calculate_nlls.py`): Requires as input a TSV file with scaffolds and decorations and outputs the same list with the NLL calculated for each one.

### Exhaustively decorating scaffolds

A special script (`sample_scaffolds.py`) to exhaustively generate a large number of decorations is supplied. This scripts can be used with both single-step and multi-step models. Notice that this script requires **both** a GPU and Spark. It works the following way:

0) A SMILES file with the scaffolds with the attachment points numbered (`[*]` -> `[*:N]`, where N is a number from 0 to the number of attachment points.
1) Generates at most `r` randomized SMILES of each scaffold (`-r` to change how many SMILES are generated at each round).
2) Samples `n` times each randomized SMILES generated in the previous step (`-n`to change the value).
3) Joins the scaffolds with the generated decorations and removes duplicates/invalids.
4) In the case of the single-step, nothing more is necessary, but in the multi-step model, a loop starting at step 1 is repeated until everything is fully decorated.
5) Everything, including the half-decorated molecules, is written down in a parquet file (or a CSV file, if the option `--output-format=csv` is used instead) for further analysis. The results have to be then extracted from the parquet/CSV file (i.e. by extracting SMILES that have the * token, for instance).

**CAUTION:** Large `n` and `r`parameters should be used for the single-step decorator model (for instance `r=2048` and `n=4096`). In the case of the multi-step model, very low values should be used instead (e.g. `r=16` and `n=32`).
**NOTICE:** A new option was added to allow using repeated randomized SMILES (`--repeated-randomized-smiles`). It is disabled by default.

Usage examples
--------------

Create the DRD2 dataset as described in the manuscript.
~~~~
(reinvent-scaffold-decorator) $> mkdir -p drd2_decorator/models
(reinvent-scaffold-decorator) $> ./slice_db.py -i training_sets/drd2.excapedb.smi.gz -u drd2_decorator/excape.drd2.hr.smi -s hr -f conditions.json.example
(reinvent-scaffold-decorator) $> ./create_randomized_smiles.py -i drd2_decorator/excape.drd2.hr.smi -o drd2_decorator/training -n 50 -d multi
~~~~
Train the DRD2 model using the training set created before.
~~~~
(reinvent-scaffold-decorator) $> ./create_model.py -i drd2_decorator/training/001.smi -o drd2_decorator/models/model.empty -d 0.2
(reinvent-scaffold-decorator) $> ./train_model.py -i drd2_decorator/models/model.empty -o drd2_decorator/models/model.trained -s drd2_decorator/training -e 50 -b 64 
~~~~
Sample one scaffold exhaustively.
~~~~
(reinvent-scaffold-decorator) $> echo "[*:0]CC=CCN1CCN(c2cccc(Cl)c2[*:1])CC1" > scaffold.smi
(reinvent-scaffold-decorator) $> spark-submit --driver-memory=8g sample_scaffolds.py -m drd2_decorator/models/model.trained.50 -i scaffold.smi -o generated_molecules.parquet -r 16 -n 16 -d multi
~~~~

**Notice**: To change it to a single-step model, the `-d single` option must be used in all cases where `-d multi` appears.
**Caution**: Spark run in local mode generally has a default of 1g of memory. This can be insufficient in some cases. That is why we use `spark-submit` to run the last script. Please change the --driver-memory=XXg to a suitable value. If you get out of memoy errors in any other script, also use the spark-submit trick.

Bugs, errors, improvements, suggestions, etc.
-----------------------------------------------

The software was thoroughly tested, although bugs may appear. Don't hesitate to contact us if you find any, or even better, send a pull request or open an issue. For other inquiries, please send an email to josep.arus@dcb.unibe.ch and we will be happy to answer you :smile:.
