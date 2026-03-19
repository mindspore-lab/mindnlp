from torch import nn, optim, LongTensor
from TransformerDecoderOnly import Transformer

def make_training_batch(sentences):
    output_batch = []
    target_batch = []
    for sentence in sentences:
        output_batch.append([tgt_vocab[n] for n in sentence[0].split()])
        target_batch.append([tgt_vocab[n] for n in sentence[1].split()])
    return LongTensor(output_batch), LongTensor(target_batch)

def make_learning_batch(sentences):
    output_batch = []
    for sentence in sentences:
        output_batch.append([tgt_vocab[n] for n in sentence.split()])
    return LongTensor(output_batch)

if __name__ == '__main__':

    sentences = [['S i want a beer', 'i want a beer E'], ['S i want a beer', 'i want a beer E']]

    # Transformer Parameters
    # Padding Should be 0
    # Unknown Should be tgt_vocab_size - 1
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6, 'U': 7}

    number_dict = {i: w for i, w in enumerate(tgt_vocab)}

    model = Transformer(len(tgt_vocab))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dec_inputs, target_batch = make_training_batch(sentences)
    
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(dec_inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    sentences = ['S P P P P'] * 10
    dec_inputs = make_learning_batch(sentences)
    for epoch in range(1):
        predict = model(dec_inputs, 16)
    predict = predict[:, :, :len(tgt_vocab) - 1].data.max(2, keepdim=True)[1]
    for sentence, output in list(zip(sentences, predict)):
        print(sentence, '->', [number_dict[n.item()] for n in output.squeeze()])