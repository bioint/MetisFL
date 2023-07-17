import argparse

from .utils import init_learner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # @stripeli the first 3-4 args are the most verbose I've ever seen :)
    # and some of them shorthands do not make sense, e.g. -e for neural engine :)
    parser.add_argument("-l", "--learner_server_entity_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="Learner server entity.")
    parser.add_argument("-c", "--controller_server_entity_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="Controller server entity.")
    parser.add_argument("-f", "--he_scheme_protobuff_serialized_hexadecimal", type=str,
                        default="",
                        help="A serialized HE Scheme protobuf message.")
    parser.add_argument("-e", "--neural_engine", type=str,
                        default="keras",
                        help="neural network training library")
    parser.add_argument("-m", "--model_dir", type=str,
                        default="",
                        help="model definition directory")
    parser.add_argument("-t", "--train_dataset", type=str,
                        default="",
                        help="train dataset filepath")
    parser.add_argument("-v", "--validation_dataset", type=str,
                        default="",
                        help="validation dataset filepath")
    parser.add_argument("-s", "--test_dataset", type=str,
                        default="",
                        help="test dataset filepath")
    parser.add_argument("-u", "--train_dataset_recipe", type=str,
                        default="",
                        help="train dataset recipe")
    parser.add_argument("-w", "--validation_dataset_recipe", type=str,
                        default="",
                        help="validation dataset recipe")
    parser.add_argument("-z", "--test_dataset_recipe", type=str,
                        default="",
                        help="test dataset recipe")
    args = parser.parse_args()
    init_learner(args)
