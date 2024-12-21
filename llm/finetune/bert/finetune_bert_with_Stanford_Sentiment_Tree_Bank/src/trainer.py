from mindnlp.core.ops import sigmoid
from mindnlp.core.nn import Tensor
from mindspore import value_and_grad, save_checkpoint

def get_accuracy_from_logits(logits, labels):
    probs = sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def evaluate(net, criterion, dataloader):
    mean_acc, mean_loss = 0, 0
    count = 0

    for data in dataloader:
        tokens_ids = data['tokens_ids']
        attn_mask = (tokens_ids != 0).long()
        label = data['label']
        logits = net(tokens_ids, attn_mask)
        mean_loss += criterion(logits.squeeze(-1), label.astype('float32')).item()
        mean_acc += get_accuracy_from_logits(logits, label)
        count += 1

    return mean_acc / count, mean_loss / count

class Trainer:
    
    def __init__(self, net, criterion, optimizer, args,
                 train_dataset, eval_dataset=None
                 ):
        self.net = net
        self.criterion = criterion
        self.opt = optimizer
        self.args = args
        self.train_dataset = train_dataset
        self.weights = self.net.trainable_params()
        self.value_and_grad = value_and_grad(self.forward_fn, None, weights=self.weights)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
        self.logits = None
    
    def forward_fn(self, tokens_ids_tensor, attn_mask, label):
        logits = self.net(tokens_ids_tensor, attn_mask)
        self.logits = logits
        loss = self.criterion(logits.squeeze(-1), label)
        return loss

    def train_single(self, tokens_ids_tensor, attn_mask, label):
        loss, grads = self.value_and_grad(tokens_ids_tensor, attn_mask, label)
        self.opt.step(grads)
        return loss

    def train(self, epochs):
        best_acc = 0
        for epoch in range(0, epochs):
            self.net.set_train(True)
            for i, data in enumerate(self.train_dataset):
                tokens_ids = data['tokens_ids']
                attn_mask = Tensor((tokens_ids != 0).long())
                label = data['label']
                
                loss = self.train_single(tokens_ids, attn_mask, label.astype('float32'))
                
                if i % self.args.print_every == 0:
                    acc = get_accuracy_from_logits(self.logits, label)
                    print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(i, epoch, loss.item(), acc))
            
            if self.run_eval:
                self.net.set_train(False)
                val_acc, val_loss = evaluate(self.net, self.criterion, self.eval_dataset)
                print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))
                if val_acc > best_acc:
                    print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
                    best_acc = val_acc
                    if self.args.save_path is not None:
                        save_checkpoint(self.net, self.args.save_path + 'best_model.ckpt')
