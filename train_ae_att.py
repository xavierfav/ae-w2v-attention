import json
import argparse

from dual_ae_att_trainer import DualAEAttTrainer


def main(config_file):
    params = json.load(open(config_file, 'rb'))
    print("Training Dual AutoEncoder with params:")
    print(json.dumps(params, separators=("\n", ": "), indent=4))
    trainer = DualAEAttTrainer(params)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dual Auto Encoder with Cross Modality Attention')
    parser.add_argument('config_file', type=str,
                        help='configuration file for the training')         
    args = parser.parse_args()

    main(args.config_file)
