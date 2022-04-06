import argparse

from trainer import Trainer6,Trainer, OPTIMIZER_LIST,Trainer1,Trainer2,Trainer3,Trainer4,Trainer5,Trainer7
from utils import init_logger, build_vocab, download_vgg_features, set_seed
from data_loader import load_data
import os

def main(args):
    init_logger()
    set_seed(args)
    download_vgg_features(args)
    build_vocab(args)

    train_dataset = load_data(args, mode="train")
    print(train_dataset)
    dev_dataset = load_data(args, mode="dev")
    test_dataset = load_data(args, mode="test")

    #print('args.trainer_id:',args.trainer_id)
    if args.trainer_id == 0:
        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
    elif args.trainer_id == 1:
        trainer = Trainer1(args, train_dataset, dev_dataset, test_dataset)
    elif args.trainer_id == 2:
        trainer = Trainer2(args, train_dataset, dev_dataset, test_dataset)
    elif  args.trainer_id == 3:
        trainer = Trainer3(args, train_dataset, dev_dataset, test_dataset)
    elif args.trainer_id == 4:
        trainer = Trainer4(args, train_dataset, dev_dataset, test_dataset)
    elif args.trainer_id == 5:
        trainer = Trainer5(args, train_dataset, dev_dataset, test_dataset)
    elif args.trainer_id == 6:
        trainer = Trainer6(args, train_dataset, dev_dataset, test_dataset)
    elif args.trainer_id == 7:
        trainer = Trainer7(args, train_dataset, dev_dataset, test_dataset)
    else:
        raise Exception('Trainer ID Error!')

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path for saving model")
    parser.add_argument("--model_file", default="model.pt", type=str, help="Path for saving model")
    parser.add_argument("--args_file", default="args.pt", type=str, help="Path for saving args")
    parser.add_argument("--trainer_id", default=0, type=int, help="Trainer ID")
    parser.add_argument("--preds_file", default="results.txt", type=str, help="Path for saving preds")
  
    parser.add_argument("--wordvec_dir", default="./wordvec", type=str, help="Path for pretrained word vector")
    parser.add_argument("--vocab_dir", default="./vocab", type=str)

    parser.add_argument("--train_file", default="train", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test", type=str, help="Test file")
    parser.add_argument("--w2v_file", default="word_vector_200d.vec", type=str, help="Pretrained word vector file")
    parser.add_argument("--img_feature_file", default="img_resnet50_features.pt", type=str, help="Filename for preprocessed image features")
    parser.add_argument("--dns_feature_file", default="dns_bert_features.pt", type=str, help="Filename for preprocessed image features")

    parser.add_argument("--max_seq_len", default=35, type=int, help="Max sentence length")
    parser.add_argument("--max_seq_len_dns", default=15, type=int, help="Max sentence length")
    parser.add_argument("--max_word_len", default=30, type=int, help="Max word length")

    parser.add_argument("--word_vocab_size", default=23204, type=int, help="Maximum size of word vocabulary")
    parser.add_argument("--char_vocab_size", default=102, type=int, help="Maximum size of character vocabulary")

    parser.add_argument("--word_emb_dim", default=200, type=int, help="Word embedding size")
    parser.add_argument("--char_emb_dim", default=30, type=int, help="Character embedding size")
    parser.add_argument("--final_char_dim", default=50, type=int, help="Dimension of character cnn output")
    parser.add_argument("--hidden_dim", default=200, type=int, help="Dimension of BiLSTM output, att layer (denoted as k) etc.")

    parser.add_argument("--kernel_lst", default="2,3,4", type=str, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")

    parser.add_argument('--seed', type=int, default=7, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation")
    parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate")
    parser.add_argument("--num_train_epochs", default=20, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--slot_pad_label", default="[pad]", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")
    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--logging_steps', type=int, default=250, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=250, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--no_w2v", action="store_true", help="Not loading pretrained word vector")

    parser.add_argument("--transformer_hidden_size", default=200, type=int, help="The initial learning rate")
    parser.add_argument("--transformer_heads", default=4, type=int, help="The initial learning rate")
    parser.add_argument("--transformer_forward_hidden_size", default=200, type=int, help="The initial learning rate")
    parser.add_argument("--transformer_dropout", default=0.1, type=float, help="The initial learning rate")

    parser.add_argument("--gpu_idx",default='0',type=str,help='gpu index')
   
    #args = parser.parse_args()

    # For VGG16 img features (DO NOT change this part)
    parser.add_argument("--num_img_region",default=49,type=int)
    parser.add_argument("--img_feat_dim",default=512,type=int)
    parser.add_argument("--dns_feat_dim",default=200,type=int)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    main(args)
