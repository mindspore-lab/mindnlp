import argparse
import time

from src.model import SentimentClassifier
from src.dataloader import SSTDataset, get_loader
from src.trainer import Trainer 
import mindspore
from mindnlp.core.nn import BCEWithLogitsLoss
from mindnlp.core.optim import Adam

def main(args):
    #Instantiating the classifier model
    print("Building model! (This might take time if you are running this for first time)")
    st = time.time()
    mindspore.set_context(device_target=args.device_target, device_id=args.device_id)
    net = SentimentClassifier(args.base_model_name_or_path, args.freeze_bert)
    print("Done in {} seconds".format(time.time() - st))

    print("Creating criterion and optimizer objects")
    st = time.time()
    criterion = BCEWithLogitsLoss()
    opti = Adam(net.trainable_params(), lr=args.lr)
    print("Done in {} seconds".format(time.time() - st))

    #Creating dataloaders
    print("Creating train and val dataloaders")
    st = time.time()
    train_set = SSTDataset(args.base_model_name_or_path, filename = 'data/SST-2/train.tsv', maxlen = args.maxlen)
    val_set = SSTDataset(args.base_model_name_or_path, filename = 'data/SST-2/dev.tsv', maxlen = args.maxlen)
    
    train_loader = get_loader(train_set, batchsize=args.batch_size)
    val_loader = get_loader(val_set, batchsize=args.batch_size, drop_remainder=False)
    print("Done in {} seconds".format(time.time() - st))

    print("Let the training begin")
    st = time.time()
    trainer = Trainer(net=net, criterion=criterion, optimizer=opti, args=args, train_dataset=train_loader, eval_dataset=val_loader)
    trainer.train(epochs=args.max_eps)
    print("Done in {} seconds".format(time.time() - st))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device_target', type = str, default = 'Ascend')
    parser.add_argument('-device_id', type = int, default = 0)
    parser.add_argument('-base_model_name_or_path', type = str, default = 'bert-base-uncased')
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type = int, default= 25)
    parser.add_argument('-batch_size', type = int, default= 32)
    parser.add_argument('-lr', type = float, default = 2e-5)
    parser.add_argument('-print_every', type = int, default= 500)
    parser.add_argument('-max_eps', type = int, default= 5)
    parser.add_argument('-save_path', type = str, default = None)
    args = parser.parse_args()

    main(args)
