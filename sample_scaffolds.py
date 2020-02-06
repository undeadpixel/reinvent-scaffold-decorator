#!/usr/bin/env python
#  coding=utf-8


import argparse
import shutil
import tempfile

import pyspark.sql as ps
import pyspark.sql.functions as psf
import pyspark.sql.types as pst

import models.model as mm
import models.actions as ma
import utils.log as ul
import utils.chem as uc
import utils.spark as us
import utils.scaffold as usc


def _cleanup_decoration(dec_smi):
    dec_mol = uc.to_mol(dec_smi)
    if not dec_mol:
        return None
    return usc.to_smiles(usc.remove_attachment_point_numbers(dec_mol))


class SampleScaffolds(ma.Action):

    cleanup_decoration_udf = psf.udf(_cleanup_decoration, pst.StringType())

    def __init__(self, model, batch_size=128, num_randomized_smiles=32, num_decorations_per_scaffold=32,
                 max_randomized_smiles_sample=10000, num_partitions=1000, decorator_type="multi", logger=None):
        ma.Action.__init__(self, logger)

        self.model = model
        self.batch_size = batch_size
        self.num_randomized_smiles = num_randomized_smiles
        self.num_decorations_per_scaffold = num_decorations_per_scaffold
        self.max_randomized_smiles_sample = max_randomized_smiles_sample
        self.num_partitions = num_partitions
        self.decorator_type = decorator_type

        self._sample_model_action = ma.SampleModel(self.model, self.batch_size, self.logger)
        self._tmp_dir = tempfile.mkdtemp(prefix="gen_lib")

    def __del__(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def run(self, initial_scaffolds):

        def _generate_randomized_scaffolds(smi, num_rand=self.num_randomized_smiles,
                                           max_rand=self.max_randomized_smiles_sample):
            mol = uc.to_mol(smi)
            randomized_scaffolds = set()
            for _ in range(max_rand):
                randomized_scaffolds.add(usc.to_smiles(mol, variant="random"))
                if len(randomized_scaffolds) == num_rand:
                    break
            return list(randomized_scaffolds)
        randomized_scaffold_udf = psf.udf(_generate_randomized_scaffolds, pst.ArrayType(pst.StringType()))
        get_attachment_points_udf = psf.udf(usc.get_attachment_points, pst.ArrayType(pst.IntegerType()))
        remove_attachment_point_numbers_udf = psf.udf(usc.remove_attachment_point_numbers, pst.StringType())

        results_df = self._initialize_results(initial_scaffolds)
        scaffolds_df = results_df.select("smiles", "scaffold", "decorations")
        i = 0
        while scaffolds_df.count() > 0:
            # generate randomized SMILES
            self._log("info", "Starting iteration #%d.", i)
            scaffolds_df = scaffolds_df.withColumn("randomized_scaffold", randomized_scaffold_udf("smiles"))\
                .select(
                    "smiles", "scaffold", "decorations",
                    psf.explode("randomized_scaffold").alias("randomized_scaffold"))\
                .withColumn("attachment_points", get_attachment_points_udf("randomized_scaffold"))\
                .withColumn("randomized_scaffold", remove_attachment_point_numbers_udf("randomized_scaffold"))\
                .persist()
            self._log("info", "Generated %d randomized SMILES from %d scaffolds.",
                      scaffolds_df.count(), scaffolds_df.select("smiles").distinct().count())

            # sample each randomized scaffold N times
            scaffolds = scaffolds_df.select("randomized_scaffold")\
                .rdd.map(lambda row: row["randomized_scaffold"]).toLocalIterator()
            self._sample_and_write_scaffolds_to_disk(scaffolds, scaffolds_df.count())
            self._log("info", "Sampled %d scaffolds.", scaffolds_df.count())

            # merge decorated molecules
            joined_df = self._join_results(scaffolds_df).persist()

            if joined_df.count() > 0:
                self._log("info", "Joined %d -> %d (valid) -> %d unique sampled scaffolds",
                          scaffolds_df.count(), joined_df.agg(psf.sum("count")).head()[0], joined_df.count())

            scaffolds_df = joined_df.join(results_df, on="smiles", how="left_anti")\
                .select("smiles", "scaffold", "decorations")\
                .where("smiles LIKE '%*%'")
            self._log("info", "Obtained %d scaffolds for next iteration.", scaffolds_df.count())

            results_df = results_df.union(joined_df)\
                .groupBy("smiles")\
                .agg(
                    psf.first("scaffold").alias("scaffold"),
                    psf.first("decorations").alias("decorations"),
                    psf.sum("count").alias("count"))\
                .persist()
            i += 1

        return results_df

    def _initialize_results(self, scaffolds):
        data = [ps.Row(smiles=scaffold, scaffold=scaffold,
                       decorations={}, count=1) for scaffold in scaffolds]
        data_schema = pst.StructType([
            pst.StructField("smiles", pst.StringType()),
            pst.StructField("scaffold", pst.StringType()),
            pst.StructField("decorations", pst.MapType(pst.IntegerType(), pst.StringType())),
            pst.StructField("count", pst.IntegerType())
        ])
        return SPARK.createDataFrame(data, schema=data_schema)

    def _sample_and_write_scaffolds_to_disk(self, scaffolds, total_scaffolds):
        def _update_file(out_file, buffer):
            for scaff, dec, _ in self._sample_model_action.run(buffer):
                out_file.write("{}\t{}\n".format(scaff, dec))

        out_file = open(self._tmp_path("sampled_decorations"), "w+")
        scaffold_buffer = []
        for scaffold in ul.progress_bar(scaffolds, total=total_scaffolds, desc="Sampling"):
            scaffold_buffer += [scaffold]*self.num_decorations_per_scaffold
            if len(scaffold_buffer) == self.batch_size*self.num_decorations_per_scaffold:
                _update_file(out_file, scaffold_buffer)
                scaffold_buffer = []

        if scaffold_buffer:
            _update_file(out_file, scaffold_buffer)
        out_file.close()

    def _join_results(self, scaffolds_df):

        def _read_rows(row):
            scaff, dec = row.split("\t")
            return ps.Row(randomized_scaffold=scaff, decoration_smi=dec)

        sampled_df = SPARK.createDataFrame(SC.textFile(self._tmp_path(
            "sampled_decorations"), self.num_partitions).map(_read_rows))

        if self.decorator_type == "single":
            processed_df = self._join_results_single(scaffolds_df, sampled_df)
        elif self.decorator_type == "multi":
            processed_df = self._join_results_multi(scaffolds_df, sampled_df)
        else:
            raise ValueError("decorator_type has an invalid value '{}'".format(self.decorator_type))

        return processed_df\
            .where("smiles IS NOT NULL")\
            .groupBy("smiles")\
            .agg(
                psf.first("scaffold").alias("scaffold"),
                psf.first("decorations").alias("decorations"),
                psf.count("smiles").alias("count"))

    def _join_results_multi(self, scaffolds_df, sampled_df):
        def _join_scaffold(scaff, dec):
            mol = usc.join(scaff, dec)
            if mol:
                return usc.to_smiles(mol)

        def _format_attachment_point(smi, num):
            smi = usc.add_first_attachment_point_number(smi, num)
            return usc.to_smiles(uc.to_mol(smi))  # canonicalize

        join_scaffold_udf = psf.udf(_join_scaffold, pst.StringType())
        format_attachment_point_udf = psf.udf(_format_attachment_point, pst.StringType())

        return sampled_df.join(scaffolds_df, on="randomized_scaffold")\
            .withColumn("decoration", format_attachment_point_udf("decoration_smi", psf.col("attachment_points")[0]))\
            .select(
                join_scaffold_udf("smiles", "decoration").alias("smiles"),
                psf.map_concat(
                    psf.create_map(psf.col("attachment_points")[0],
                                   SampleScaffolds.cleanup_decoration_udf("decoration")),
                    "decorations",
                ).alias("decorations"),
                "scaffold")

    def _join_results_single(self, scaffolds_df, sampled_df):
        def _join_scaffold(scaff, decs):
            mol = usc.join_joined_attachments(scaff, decs)
            if mol:
                return usc.to_smiles(mol)
        join_scaffold_udf = psf.udf(_join_scaffold, pst.StringType())

        def _create_decorations_map(decorations_smi, attachment_points):
            decorations = decorations_smi.split(usc.ATTACHMENT_SEPARATOR_TOKEN)
            return {idx: _cleanup_decoration(dec) for dec, idx in zip(decorations, attachment_points)}
        create_decorations_map_udf = psf.udf(_create_decorations_map, pst.MapType(pst.IntegerType(), pst.StringType()))

        return sampled_df.join(scaffolds_df, on="randomized_scaffold")\
            .select(
                join_scaffold_udf("randomized_scaffold", "decoration_smi").alias("smiles"),
                create_decorations_map_udf("decoration_smi", "attachment_points").alias("decorations"),
                "scaffold")

    def _tmp_path(self, file_name):
        return "{}/{}".format(self._tmp_dir, file_name)


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Generates large amounts of molecules from a set of scaffolds.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--input-scaffold-path", "-i",
                        help="Path to the input file with scaffolds in SMILES notation.", type=str, required=True)
    parser.add_argument("--output-path", "-o",
                        help="Path to the output file or directory (see --output-format option for more information).", type=str, required=True)
    parser.add_argument("--batch-size", "-b",
                        help="Batch size (beware GPU memory usage) [DEFAULT: 128]", type=int, default=128)
    parser.add_argument("--num-randomized-smiles", "-r",
                        help="Number of randomized SMILES to use in every stage of the \
                            decoration process. [DEFAULT: 32]",
                        type=int, default=32)
    parser.add_argument("--num-decorations-per-scaffold", "-n",
                        help="Number of times to sample the model for a given \
                            randomized SMILES scaffold. [DEFAULT: 32]",
                        type=int, default=32)
    parser.add_argument("--num-partitions", "--np",
                        help="Number of Spark partitions to use (leave it if you don't know what it means) \
                            [DEFAULT: 1000]",
                        type=int, default=1000)
    parser.add_argument("--decorator-type", "-d",
                        help="Type of decorator TYPES=(single, multi) [DEFAULT: multi].",
                        type=str, default="multi")
    parser.add_argument("--output-format", "--of",
                        help="Format of the output FORMATS=(parquet,csv) [DEFAULT: parquet].",
                        type=str, default="parquet")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    model = mm.DecoratorModel.load_from_file(args.model_path, mode="eval")
    input_scaffolds = list(uc.read_smi_file(args.input_scaffold_path))

    sample_scaffolds = SampleScaffolds(model, num_randomized_smiles=args.num_randomized_smiles,
                                       num_decorations_per_scaffold=args.num_decorations_per_scaffold,
                                       decorator_type=args.decorator_type, batch_size=args.batch_size,
                                       num_partitions=args.num_partitions, logger=LOG)

    results_df = sample_scaffolds.run(input_scaffolds)

    if args.output_format == "parquet":
        results_df.write.parquet(args.output_path)
    else:
        results_df.toPandas().to_csv(args.output_path)


LOG = ul.get_logger(name="sample_scaffolds")
SPARK, SC = us.SparkSessionSingleton.get("sample_scaffolds")
if __name__ == "__main__":
    main()
