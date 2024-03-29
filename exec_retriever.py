import json
import argparse
import contextlib
import sh
import subprocess
import pathlib
from allennlp.commands.train import train_model
from allennlp.common import Params
# from smbop.dataset_readers.spider_basic_pkl import SmbopSpiderDatasetReader
from smbop.dataset_readers.spider_retriever_nn import SmbopSpiderRetrieverDatasetReader
from smbop.dataset_readers.spider_retriever_nn_val import SmbopSpiderRetrieverValDatasetReader
# from smbop.vocabulary.vocab_retriever import RetriverSpiderVocabulary
# from smbop.models.smbop_vanilla import SmbopParser
# from smbop.models.cbr_smbop import CBRSmbopParser
# from smbop.models.smbop_pretrain import SmbopSimPretrained
from smbop.models.retriever_soft_no_leafs import SmbopSimPretrained
# from smbop.models.retriever_soft import SmbopSimPretrained
from smbop.modules.relation_transformer import RelationTransformer
from smbop.modules.lxmert import LxmertCrossAttentionLayer
from smbop.data_loaders.data_loader_retriever import RetriverMultiProcessDataLoader
import namegenerator
import os
import torch
def to_string(value):
    if isinstance(value, list):
        return [to_string(x) for x in value]
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value)


def run():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--name", nargs="?")
    parser.add_argument("--force", action="store_true",
                        help="""If True, we will overwrite the serialization
                                directory if it already exists.""")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--recover", action="store_true",
                        help= """If True, we will try to recover a training run
                                 from an existing serialization directory. 
                                 This is only intended for use when something 
                                 actually crashed during the middle of a run. 
                                 For continuing training a model on new data,
                                  see Model.from_archive.""")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--detect_anomoly", action="store_true") #IDK
    parser.add_argument("--profile", action="store_true") #IDK: some sort of debugging funct
    parser.add_argument("--is_oracle", action="store_true")
    parser.add_argument("--tiny_dataset", action="store_true")
    parser.add_argument("--load_less", action="store_true")
    parser.add_argument("--cntx_rep", action="store_true")
    parser.add_argument("--cntx_beam", action="store_true")
    parser.add_argument("--disable_disentangle_cntx", action="store_true")
    parser.add_argument("--disable_cntx_reranker", action="store_true")
    parser.add_argument("--disable_value_pred", action="store_true")
    parser.add_argument("--disable_use_longdb", action="store_true")
    parser.add_argument("--uniquify", action="store_true")
    parser.add_argument("--use_bce", action="store_true")
    parser.add_argument("--tfixup", action="store_true")
    parser.add_argument("--train_as_dev", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--disable_utt_aug", action="store_true")
    parser.add_argument("--should_rerank", action="store_true")
    parser.add_argument("--use_treelstm", action="store_true")
    parser.add_argument("--disable_db_content", action="store_true",
                        help="Run with this argument (once) before pre-proccessing to reduce the pre-proccessing time by half \
                         This argument causes EncPreproc to not perform IR on the largest tables. ")
    parser.add_argument("--lin_after_cntx", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--rat_layers", type=int, default=8)
    parser.add_argument("--beam_size", default=30, type=int)
    parser.add_argument("--base_dim", default=32, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--beam_encoder_num_layers", default=1, type=int)
    parser.add_argument("--tree_rep_transformer_num_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--rat_dropout", default=0.2, type=float)
    parser.add_argument("--lm_lr", default=3e-6, type=float)
    parser.add_argument("--lr", type=float, default=0.000186)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--grad_acum", default=4, type=int)
    parser.add_argument("--max_steps", default=60000, type=int)
    parser.add_argument("--power", default=0.5, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--grad_clip", default=-1, type=float)
    parser.add_argument("--grad_norm", default=-1, type=float)

    default_dict = {k.option_strings[0][2:]: k.default for k in parser._actions}
    args = parser.parse_args()
    diff = "_".join(
        [
            f"{key}{value}"
            for key, value in vars(args).items()
            if (key != "name" and value != default_dict[key])
        ]
    ) #vars which differ from default

    ext_vars = {}
    for key, value in vars(args).items():
        if key.startswith("disable"):
            new_key = key.replace("disable_", "")
            ext_vars[new_key] = to_string(not value)
        else:
            ext_vars[key] = to_string(value)
    print(ext_vars) #just a toggle of disabled variables
    default_config_file = "configs/retriever.jsonnet" # NOTE: defaults.jsonnet was original value

    overrides_dict = {}

    if args.profile:
        overrides_dict["trainer"]["num_epochs"] = 1

    experiment_name_parts = []
    experiment_name_parts.append(namegenerator.gen())
    if diff:
        experiment_name_parts.append(diff)
    if args.name:
        experiment_name_parts.append(args.name)

    experiment_name = "_".join(experiment_name_parts)
    print(f"experiment_name: {experiment_name}")
    ext_vars["experiment_name"] = experiment_name
    overrides_json = json.dumps(overrides_dict)
    settings = Params.from_file(
        default_config_file,
        ext_vars=ext_vars,
        params_overrides=overrides_json,
    )

    print(settings)
    prefix = ""
    # prefix = "/home/ohadr/"
    prefix = "/mnt/infonas/data/alirehan/smbop/try_train/"


    assert not pathlib.Path(f"{prefix}experiments/{experiment_name}").exists()

#     sh.ln("-s", f"{prefix}/experiments/{experiment_name}", f"experiments/{experiment_name}")
    pathlib.Path(f"backup").mkdir(exist_ok=True)
    pathlib.Path(f"cache").mkdir(exist_ok=True)
    # pathlib.Path(f"experiments/{experiment_name}").mkdir(exist_ok=True)

    # subprocess.check_call(
    #     f"git ls-files | tar Tzcf - backup/{experiment_name}.tgz", shell=True
    # )

    if args.profile:
        pass
    else:
        cntx = contextlib.nullcontext()

    with cntx:
        train_model(
            params=settings,
            serialization_dir=f"{prefix}experiments/{experiment_name}",
            recover=args.recover,
            force=True,
        )


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICE']="2"
    # os.environ['NVIDIA_VISIBLE_DEVICE']="2"
    print(torch.cuda.get_device_name(0))
    run()
