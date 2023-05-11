
quanjiao2b = ['……', '。。。', '。', '，', '；', '：', '？', '！', '“', '”', '‘', '’' '（', '）', '【', '】', '、']
banjiao = ['...', '...', '.', ',', ';', ':', '?', '!', '"', '"', ",", ",", "(", ")", '[', ']', ',']

quanjiao = ['……', '。。。', '。', '，', '；', '：', '？', '！', '（', '）', '【', '】']
banjiao2q = ['...', '...', '.', ',', ';', ':', '?', '!', "(", ")", '[', ']', ',']


def repl(data, fromjiao, tojiao):
    assert len(fromjiao) == len(tojiao)
    for i, j in zip(fromjiao, tojiao):
        data = data.replace(i, j)
    return data


def process(line):
    new_line = line.replace('\n', ' ')
    puncs = [',', '.', ';', ':', '"', "'", '?', '!']
    p1 = [',', '.', ';', ':', '?', '!']
    p2 = ["'"]
    p3 = []
    for _ in range(5):
        for p in p1:
            new_line = new_line.replace(' ' + p + ' ', p)
            new_line = new_line.replace(p + ' ', p)
            new_line = new_line.replace(' ' + p, p)

    for p in p1:
        new_line = new_line.replace(p, p + ' ')

    new_line = new_line.replace('. . . ', '... ')

    wrong_samples = []
    for i in range(1, len(new_line) - 2):
        if new_line[i] == "'" and new_line[i+1].isalpha() and new_line[i - 1] == ' ' and new_line[i + 2] == ' ':
            j = i - 2
            while j >= 1 and new_line[j] == ' ':
                j -= 1
            wrong_samples.append(new_line[j: i + 3])

    wrong_samples.sort(key=lambda x: len(x), reverse=True)
    for w in wrong_samples:
        new_line = new_line.replace(w, w[0] + w[-3: ])
    new_line = new_line.replace(" n't", "n't")
    for k in range(len(new_line) - 1, -1, -1):
        if new_line[k] != ' ':
            new_line = new_line[: k + 1]
            break
    return new_line


def en_cleaning(data):
    d = repl(data, quanjiao2b, banjiao)
    d = process(d)
    new_d = d.replace('  ', ' ')
    while new_d != d:
        d = new_d
        new_d = d.replace('  ', ' ')
    return new_d
